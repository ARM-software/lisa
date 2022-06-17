# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

r"""
This module provides classes to build kernel modules from source on the fly.

Here is an example of such module::

    import time

    from lisa.target import Target
    from lisa.trace import DmesgCollector
    from lisa._kmod import KmodSrc

    from lisa.utils import setup_logging
    setup_logging()

    target = Target(
        kind='linux',
        name='my_board',
        host='192.158.1.38',
        username='root',
        password='root',
        lazy_platinfo=True,
        kernel_src='/path/to/kernel/tree/',
        kmod_build_env='alpine',
        # kmod_make_vars={'CC': 'clang'},
    )

    # Example module from: https://tldp.org/LDP/lkmpg/2.6/html/x279.html
    code = r'''
    /*
    *  hello-4.c - Demonstrates module documentation.
    */
    #include <linux/module.h> /* Needed by all modules */
    #include <linux/kernel.h> /* Needed for KERN_INFO */
    #include <linux/init.h>   /* Needed for the macros */
    #define DRIVER_AUTHOR "XXX"
    #define DRIVER_DESC   "A sample driver"

    static int __init init_hello(void)
    {
        printk(KERN_INFO "Hello, world\n");
        return 0;
    }

    static void __exit cleanup_hello(void)
    {
        printk(KERN_INFO "Goodbye, worldn");
    }

    module_init(init_hello);
    module_exit(cleanup_hello);

    /*
    *  You can use strings, like this:
    */

    /*
    * Get rid of taint message by declaring code as GPL.
    */
    MODULE_LICENSE("GPL");

    /*
    * Or with defines, like this
    */
    MODULE_AUTHOR(DRIVER_AUTHOR);    /* Who wrote this module? */
    MODULE_DESCRIPTION(DRIVER_DESC); /* What does this module do */
    '''

    # This object represents the kernel sources, and needs to be turned into a
    # DynamicKmod to be compiled and run.
    src = KmodSrc({'hello.c': code})

    # Create a DynamicKmod from the target and the module sources.
    kmod = target.get_kmod(src=src)

    # Collect the dmesg output while running the module
    dmesg_coll = DmesgCollector(target, output_path='dmesg.log')

    # kmod.run() will compile the module, install it and then uninstall it at the
    # end of the "with" statement.
    with dmesg_coll, kmod.run():
        time.sleep(1)

    for entry in dmesg_coll.entries:
        print(entry)
"""

import abc
import urllib.request
import urllib.parse
from pathlib import Path, PurePosixPath
import tempfile
import tarfile
import gzip
import bz2
import lzma
import os
import io
import shutil
import contextlib
import subprocess
import copy
import re
import functools
import bisect
import threading
import hashlib
import itertools
import logging
import datetime
import pwd
import glob
import collections
import collections.abc
import hashlib
from operator import itemgetter
from shlex import quote
from io import BytesIO

from elftools.elf.elffile import ELFFile

from devlib.target import KernelVersion, TypedKernelConfig, KernelConfigTristate
from devlib.host import LocalConnection
from devlib.exception import TargetStableError

from lisa.utils import nullcontext, Loggable, LISA_CACHE_HOME, checksum, DirCache, chain_cm, memoized, LISA_HOST_ABI, subprocess_log, SerializeViaConstructor
from lisa._assets import ASSETS_PATH, HOST_PATH
from lisa._unshare import ensure_root
import lisa._git as git

_ALPINE_ROOTFS_URL = 'https://dl-cdn.alpinelinux.org/alpine/v{minor}/releases/{arch}/alpine-minirootfs-{version}-{arch}.tar.gz'

def _any_abi_to_kernel_arch(abi):
    return {
        'armeabi': 'arm',
        'armv7': 'arm',
        'aarch64': 'arm64',
    }.get(abi, abi)


def _url_path(url):
    return PurePosixPath(
        urllib.parse.unquote(
            urllib.parse.urlparse(url).path
        )
    )


def _subprocess_log(*args, env=None, extra_env=None, **kwargs):
    if env is None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k in ('PATH', 'USER', 'TERM')
        }

    env.update(extra_env or {})
    return subprocess_log(*args, **kwargs, env=env)


@contextlib.contextmanager
def _make_chroot(make_vars, bind_paths=None, alpine_version='3.15.0', overlay_backend=None):
    """
    Create a chroot folder ready to be used to build a kernel.
    """
    logger = logging.getLogger(f'{__name__}.alpine_chroot')

    def mount_binds(chroot, bind_paths, mount=True):
        for src, dst in bind_paths.items():
            dst = Path(dst).resolve()
            dst = (chroot / dst.relative_to('/')).resolve()
            # This will be unmounted by the destroy script
            if mount:
                dst.mkdir(parents=True, exist_ok=True)
                cmd = ['mount', '--bind', '--', src, dst]
            else:
                cmd = ['umount', '-n', '--', dst]
            _subprocess_log(cmd, logger=logger, level=logging.DEBUG)

    def populate(key, path, init_cache=True):
        version, arch, packages, use_qemu = key
        path = path.resolve()

        # Packages have already been installed, so we can speed things up a
        # bit
        if init_cache:
            packages = packages.split(' ')

            _version = version.split('.')
            minor = '.'.join(_version[:-1])
            url = _ALPINE_ROOTFS_URL.format(
                minor=minor,
                arch=arch,
                version=version,
            )

            with tempfile.NamedTemporaryFile(dir=path) as f:
                tar_path = Path(f.name).resolve()
                logger.info(f'Setting up Alpine chroot from {url} -> {tar_path}')

                with urllib.request.urlopen(url) as url, open(tar_path, 'wb') as f:
                    shutil.copyfileobj(url, f)

                with tarfile.open(tar_path, 'r') as f:
                    f.extractall(path=path)
        else:
            packages = []

        shutil.copy('/etc/resolv.conf', path / 'etc' / 'resolv.conf')

        if packages:
            cmd = _make_chroot_cmd(path, ['apk', 'add', *packages])
            _subprocess_log(cmd, logger=logger, level=logging.DEBUG)

    packages = [
        'bash',
        'binutils',
        'coreutils',
        'diffutils',
        'make',
        'file',
        'gawk',
        'sed',
        'musl-dev',
        'elfutils-dev',
        'gmp-dev',
        'libffi-dev',
        'openssl-dev',
        'linux-headers',
        'musl',
        'bison',
        'flex',
        'python3',

        # TODO: As of october 2021 for some reason, the kernel still needs GCC
        # to build some tools even when compiling with clang
        'gcc',
    ]
    make_vars = make_vars or {}

    try:
        cc = make_vars['CC']
    except KeyError:
        cc = 'gcc'

    if cc == 'clang':
        packages.extend([
            'lld',
            'llvm',
        ])
    packages.append(cc)

    devlib_arch = make_vars.get('ARCH', LISA_HOST_ABI)

    use_qemu = (
        devlib_arch != LISA_HOST_ABI and
        # Since clang binaries support cross compilation without issues,
        # there is no need to use QEMU that will slow everything down.
        make_vars.get('CC') != 'clang'
    )

    qemu_arch = {
        'arm64': 'aarch64',
        'armeabi': 'arm',
        'armv7': 'arm',
    }.get(devlib_arch, devlib_arch)
    binfmt_path = Path('/proc/sys/fs/binfmt_misc/', f'qemu-{qemu_arch}')
    if use_qemu and not binfmt_path.exists():
        raise ValueError(f'Alpine chroot is setup for {qemu_arch} architecture but QEMU userspace emulation is not installed on the host (missing {binfmt_path})')

    if use_qemu:
        chroot_arch = devlib_arch
    else:
        chroot_arch = LISA_HOST_ABI

    alpine_arch = {
        'arm64': 'aarch64',
        'armeabi': 'armv7',
    }.get(chroot_arch, chroot_arch)

    # Add LISA static binaries inside the chroot
    bind_paths = {
        **dict(bind_paths or {}),
        str((Path(ASSETS_PATH) / 'binaries' / devlib_arch).resolve()): '/usr/local/bin/'
    }

    dir_cache = DirCache(
        category='alpine_chroot',
        populate=populate,
    )

    key = (
        alpine_version,
        alpine_arch,
        ' '.join(sorted(packages)),
        use_qemu,
    )
    cache_path = dir_cache.get_entry(key)
    with _overlay_folders([cache_path], backend=overlay_backend) as path:
        # We need to "repopulate" the overlay in order to get a working
        # system with /etc/resolv.conf etc
        try:
            populate(key, path, init_cache=False)
            mount_binds(path, bind_paths)
            yield path
        finally:
            mount_binds(path, bind_paths, mount=False)


def _make_chroot_cmd(chroot, cmd):
    chroot = Path(chroot).resolve()
    cmd = ' '.join(map(quote, map(str, cmd)))
    # Source /etc/profile to get sane defaults for e.g. PATH. Otherwise, we
    # just inherit it from the host, which is broken.
    cmd = f'source /etc/profile && exec {cmd}'
    return ['chroot', chroot, 'sh', '-c', cmd]


@contextlib.contextmanager
def _overlay_folders(lowers, upper=None, backend=None, copy_filter=None):
    """
    Overlay folders on top of each other.

    :param lowers: List of read-only lower layers. The end of the list takes
        precedence.
    :type lowers: list(str)

    :param upper: Read-write upper layer taking all the changes made to the
        mount point. If left out, a throw-away upper layer will be used.
    :type upper: str or None

    :param backend: Backend to use, one of:

        * ``overlayfs``: Uses Linux overlayfs mounts. This is the fastest and
          most space efficient method.
        * ``copy``: This uses plain copies to simulate overlayfs. Note that the
          simulation is not entirely transparent, as a higher layer is not able
          to hide files in lower layers like it can do with overlayfs and
          whiteout files.
        * ``None``: defaults to ``overlayfs``.
    :type backend: str or None
    """
    logger = logging.getLogger(f'{__name__}.overlay')
    backend = KernelTree._resolve_overlay_backend(backend)

    def make_dir(root, name):
        path = Path(root) / name
        path.mkdir(parents=True)
        return path.resolve()

    with tempfile.TemporaryDirectory() as temp:
        mount_point = make_dir(temp, 'overlaid')

        # Work folder has to be in the same filesystem as the upper dir
        if upper:
            @contextlib.contextmanager
            def dirs_cm():
                with tempfile.TemporaryDirectory(
                    # We cannot use a subfolder of "upper" to host "work" so we
                    # have to use the parent folder.
                    dir=upper.parent,
                    prefix='.overlayfs_work_'
                ) as work:
                    yield dict(
                        work=Path(work),
                        upper=Path(upper),
                    )
        else:
            @contextlib.contextmanager
            def dirs_cm():
                yield dict(
                    work=make_dir(temp, 'work'),
                    upper=make_dir(temp, 'upper'),
                )

        @contextlib.contextmanager
        def do_mount(dirs):
            dirs['lower'] = ':'.join(map(str, reversed(list(lowers))))
            cmd = ['mount', '-t', 'overlay', 'overlay', '-o', 'lowerdir={lower},workdir={work},upperdir={upper}'.format(**dirs), '--', mount_point]
            _subprocess_log(cmd, logger=logger, level=logging.DEBUG)

            try:
                yield mount_point
            finally:
                # Use lazy unmount, so it will not fail if it still in use for
                # some reason. That said, all supporting folders are going to
                # be removed so an external user working outside of the "with"
                # statement will have issues, which is expected (and not
                # supported).
                _subprocess_log(
                    ['umount', '-nl', '--', mount_point],
                    logger=logger,
                    level=logging.DEBUG
                )

        if copy_filter is None:
            copy_filter = lambda src, dst: True

        @contextlib.contextmanager
        def do_copy(dirs):
            def _python_copytree(src, dst):
                base_src = Path(src)
                base_dst = Path(dst)
                def copy_file(src, dst):
                    if not copy_filter(
                        src=Path(src).relative_to(base_src),
                        dst=Path(dst).relative_to(base_dst)
                    ):
                        return dst

                    if os.path.islink(src):
                        if os.path.lexists(dst):
                            os.unlink(dst)
                        linkto = os.readlink(src)
                        os.symlink(linkto, dst)
                        shutil.copystat(src, dst, follow_symlinks=False)
                        return dst
                    else:
                        try:
                            dst_mtime = os.path.getmtime(dst)
                        except OSError as e:
                            if not isinstance(e, FileNotFoundError):
                                os.remove(dst)
                            return shutil.copy2(src=src, dst=dst)
                        else:
                            src_mtime = os.path.getmtime(src)
                            # Only copy files that have been modified more recently
                            if src_mtime > dst_mtime:
                                os.remove(dst)
                                return shutil.copy2(src=src, dst=dst)
                            else:
                                return dst
                shutil.copytree(
                    src=str(src),
                    dst=str(dst),
                    # Make symlinks go through copy_function
                    symlinks=False,
                    dirs_exist_ok=True,
                    copy_function=copy_file,
                )

            def _rsync_copytree(src, dst):
                subprocess.check_call(['rsync', '-au', '--', f'{src}/', dst])

            def _copytree(src, dst):
                try:
                    _rsync_copytree(src, dst)
                # rsync not installed
                except FileNotFoundError:
                    logger.debug('rsync not installed, falling back on python copy')
                    _python_copytree(src, dst)

            logger.debug(f'Copying trees instead of overlayfs for {mount_point}')
            for src in lowers:
                _copytree(src=src, dst=mount_point)

            try:
                yield mount_point
            finally:
                # If the user selected a custom upper layer, sync back the
                # result in it
                if upper:
                    shutil.rmtree(upper)
                    shutil.move(src=mount_point, dst=upper)

        if backend == 'overlayfs':
            action = do_mount
        elif backend == 'copy':
            action = do_copy
        else:
            raise ValueError(f'Unknwon overlay backend "{backend}"')

        with dirs_cm() as dirs:
            with action(dirs) as mnt:
                yield mnt


class OverlayResource(abc.ABC):
    """
    Resource to be applied as an overlay in an existing folder.
    """
    @abc.abstractmethod
    def write_to(self, dst):
        """
        Write the resource to the ``dst`` path.
        """
        pass

    @abc.abstractmethod
    def _get_checksum(self):
        """
        Return the checksum of the resource.
        """
        pass


class _FileOverlayBase(OverlayResource):
    """
    :meta public:

    Base class for file overlays.
    """
    pass


class FileOverlay(_FileOverlayBase):
    """
    Overlay representing a file content.
    """
    @classmethod
    def from_content(cls, content):
        """
        Build the file from its ``content``.
        """
        return _ContentFileOverlay(content)

    @classmethod
    def from_path(cls, path, decompress=False):
        """
        Build the file from an existing path.

        :param decompress: If ``True``, the file will be decompressed according
            to its extension. E.g. an ``.gz`` file would be inferred as gzip
            and decompressed. If ``False``, the extension is ignored.
        :type decompress: bool
        """
        if decompress:
            return _CompressedPathFileOverlay(path)
        else:
            return _PathFileOverlay(path)


class _PathOverlayBase(_FileOverlayBase):
    """
    :meta public:

    Base class for path-based overlays.
    """
    # This is racy with write_to(), but we are not trying to make something
    # really secure here, we just want to compute a unique token to be used as
    # a cache key
    def _get_checksum(self):
        with open(self.path, 'rb') as f:
            check = checksum(f, 'sha256')
        return f'{self.__class__.__name__}-{check}'

    def __str__(self):
        return str(self.path)


class _PathFileOverlay(_PathOverlayBase):
    def __init__(self, path):
        self.path = path

    def write_to(self, dst):
        shutil.copy2(self.path, dst)


class _CompressedPathFileOverlay(_PathOverlayBase):
    _OPEN_FUNCTIONS = {
        '.gz': gzip.open,
        '.xz': lzma.open,
        '.bz': bz2.open,
    }

    def __init__(self, path):
        path = Path(path)
        try:
            open_f = self._OPEN_FUNCTIONS[path.suffix]
        except KeyError:
            raise ValueError(f'Could not detect compression format of "{path}". Tried {", ".join(cls._OPEN_FUNCTIONS.keys())}')
        else:
            self.open_f = open_f
            self.path = path

    def write_to(self, dst):
        with self.open_f(self.path) as src, open(dst, 'wb') as dst:
            shutil.copyfileobj(src, dst)


class _ContentFileOverlay(_FileOverlayBase):
    def __init__(self, content):
        self.content = content

    def write_to(self, dst):
        with open(dst, 'wb') as f:
            f.write(self.content)

    def _get_checksum(self):
        check = checksum(io.BytesIO(self.content), 'sha256')
        return f'{self.__class__.__name__}-{check}'


class TarOverlay(_PathOverlayBase):
    """
    The ``__init__`` constructor is considered private. Use factory classmethod
    to create instances.
    """
    def __init__(self, path):
        self.path = path

    @classmethod
    def from_path(cls, path):
        """
        Build the overlay from the ``path`` to an existing tar archive.
        """
        return cls(path)

    def write_to(self, dst):
        with tarfile.open(self.path) as tar:
            tar.extractall(dst)


class KernelTree(Loggable, SerializeViaConstructor):
    """
    :param path_cm: Context manager factory expected to return a path to a
        prepared kernel tree.
    :type path_cm: collections.abc.Callable

    :param make_vars: Variables passed on ``make`` command line when preparing
        the kernel tree.
    :type make_vars: dict(str, str)

    :param build_env: Build environment to use. Can be one of:

        * ``alpine``: Alpine linux chroot, providing a controlled
          environment
        * ``host``: No specific env is setup, whatever the host is using will
          be picked.
        * ``None``: defaults to ``host``.
    :type build_env: str or None

    :param overlay_backend: Backend used to create folder overlays. One of:

        * ``overlayfs``: Use overlayfs Linux filesystem. This is the fastest
          and the recommanded option.
        * ``copy``: Use plain folder copies. This can be used as an alternative
          if overlayfs cannot be used for some reason.
        * ``None``: default to ``overlayfs``.
    """

    # Preserve checksum attribute when serializing, as it will allow hitting
    # the module cache without actually setting up the kernel tree in many
    # cases.
    _SERIALIZE_PRESERVED_ATTRS = {'checksum'}

    _KERNEL_ARCHIVE_URL_TEMPLATE = 'https://cdn.kernel.org/pub/linux/kernel/v{main_number}.x/linux-{version}.tar.xz'

    # We are only really interested in clang starting from version 13,
    # when the "musttail" return attribute was introduced.
    # On top of that, the kernel does not handle clang < 10.0.1
    _MIN_CLANG_VERSION = 11

    def __init__(self, path_cm, make_vars, build_env=None, overlay_backend=None):
        self._make_path_cm = path_cm
        self.build_env = self._resolve_build_env(build_env)
        self.make_vars = make_vars or {}
        self.overlay_backend = self._resolve_overlay_backend(overlay_backend)
        self._path_cm = None
        self.path = None
        self.checksum = None

    @staticmethod
    def _resolve_build_env(build_env):
        return build_env or 'host'

    @staticmethod
    def _resolve_overlay_backend(overlay_backend):
        return overlay_backend or 'overlayfs'

    def _to_spec(self):
        return dict(
            path=self.path,
            checksum=self.checksum,
        )

    def _update_spec(self, spec):
        def update(x):
            val = spec.get(x)
            if val is not None:
                setattr(self, x, val)
        if spec:
            for attr in ('path', 'checksum'):
                update(attr)

    # It is expected that the same object can be used more than once, so
    # __enter__ and __exit__ must not do anything destructive.
    def __enter__(self):
        cm = self._make_path_cm()
        spec = cm.__enter__()
        assert 'path' in spec
        self._update_spec(spec)
        self._path_cm = cm
        return self

    def __exit__(self, *args, **kwargs):
        # Reset the path as it cannot be used outside the with statement but
        # not the checksum, since it could still be reused to hit the cache
        self.path = None
        try:
            ret = self._path_cm.__exit__(*args, **kwargs)
        finally:
            self._path_cm = None
        return ret

    _URL_CACHE = {}

    @classmethod
    def _open_url(cls, version):
        url, response = cls._get_url_response(version)
        if response is None:
            response = urllib.request.urlopen(url)
        return response

    @classmethod
    def _get_url(cls, version):
        url, response = cls._get_url_response(version)
        with (response or nullcontext()):
            return url

    @classmethod
    def _get_url_response(cls, version):
        def replace_None(tuple_):
            return tuple(
                0 if x is None else x
                for x in tuple_
            )

        def make_url(parts):
            # Remove trailing 0 as this seems to be the pattern followed by
            # cdn.kernel.org URLs
            parts = parts if parts[-1] else parts[:-1]

            return cls._KERNEL_ARCHIVE_URL_TEMPLATE.format(
                main_number=parts[0],
                version='.'.join(map(str, parts)),
            )

        @functools.lru_cache
        def get_available_versions():
            url = make_url(orig_version.parts)
            parsed = urllib.parse.urlparse(url)
            index_url = parsed._replace(path=str(_url_path(url).parent)).geturl()
            with urllib.request.urlopen(index_url) as url_f:
                html = url_f.read()
            files = re.findall(rb'href="linux-(.*)\.tar\.xz"', html)
            return sorted(
                (
                    replace_None(
                        KernelVersion(name.decode()).parts
                    )
                    for name in files
                ),
            )

        def decrement_version(parts):
            parts = replace_None(parts)
            versions = get_available_versions()
            i = bisect.bisect(versions, parts) - 2
            try:
                return versions[i]
            except IndexError:
                raise ValueError(f'Could not find any kernel tarball for version {parts}')

        orig_version = version
        parts = version.parts
        logger = cls.get_logger()

        try:
            url = cls._URL_CACHE[str(version)]
        except KeyError:
            while True:
                url = make_url(parts)
                logger.debug(f'Trying to fetch {url} for kernel version {orig_version} ...')

                try:
                    response = urllib.request.urlopen(url)
                except urllib.error.HTTPError as e:
                    # Maybe this is a development kernel and no release has been
                    # done for that version yet, keep trying with lower versions
                    if e.code == 404:
                        try:
                            parts = decrement_version(parts)
                        except ValueError:
                            raise ValueError('Cannot fetch any tarball matching {orig_version}')
                else:
                    cls._URL_CACHE[str(version)] = response.url
                    return (url, response)
        else:
            return (url, None)


    @classmethod
    def _prepare_tree(cls, path, make_vars, build_env, apply_overlays, overlay_backend):
        logger = cls.get_logger()
        _make_vars = [
            f'{name}={val}'
            for name, val in sorted((make_vars or {}).items())
            if val is not None
        ]
        nr_cpus = int(os.cpu_count() * 1.5)


        def make(*targets):
            return ['make', f'-j{nr_cpus}', '-C', path, '--', *_make_vars, *targets]

        cmds = [
            # On non-host build env, we need to clean first, as binaries compiled
            # in e.g. scripts/ will probably not work inside the Alpine container,
            # since they would be linked against shared libraries on the host
            # system.
            #
            # On host build env, due to this bug:
            # https://lore.kernel.org/all/YfK18x%2FXrYL4Vw8o@syu-laptop/t/#md877c45455918f8c661dc324719b91a9906dc7a3
            # We need to get rid of vmlinux file in order to prevent the kernel
            # module build from generating split BTF information.
            #
            # On top of that, a user-provided kernel tree with modules_prepare
            # already run would lead to not having e.g. some binaries stored in
            # the overlay. All would be well until the user does a manual
            # mrproper, at which point we are left hanging since we would hit
            # the cache, but would not be ready. "make modules" would probably
            # fix that but we could still have possibly issues with different
            # toolchains etc.
            make('mrproper'),
            make('olddefconfig', 'modules_prepare'),
        ]

        bind_paths = {path: path}

        if build_env == 'alpine':
            @contextlib.contextmanager
            def cmd_cm(cmds):
                with _make_chroot(bind_paths=bind_paths, make_vars=make_vars, overlay_backend=overlay_backend) as chroot:
                    yield [
                        _make_chroot_cmd(chroot, cmd) if cmd else None
                        for cmd in cmds
                    ]
        else:
            cmd_cm = lambda cmds: nullcontext(cmds)

        with cmd_cm(cmds) as _cmds:
            pre, post = _cmds
            logger.info(f'Preparing kernel tree for modules')

            if pre is not None:
                _subprocess_log(pre, logger=logger, level=logging.DEBUG)

            # Apply the overlays before running make, so that it sees the
            # correct headers and conf etc
            apply_overlays()

            _subprocess_log(post, logger=logger, level=logging.DEBUG)
            # Re-apply the overlays, since we could have overwritten important
            # things, such as include/linux/vermagic.h
            apply_overlays()


    @classmethod
    def _process_make_vars(cls, build_env, make_vars, abi=None):
        env = {
            k: str(v)
            for k, v in (
                (k, os.getenv(k)) for k in {
                    'CROSS_COMPILE',
                    'ARCH',
                }
            )
            if v is not None
        }

        make_vars = {
            **env,
            **dict(make_vars or {})
        }
        if abi is None:
            abi = make_vars.get('ARCH', LISA_HOST_ABI)

        arch = _any_abi_to_kernel_arch(abi)
        make_vars['ARCH'] = arch
        make_vars, cc = cls._resolve_toolchain(abi, make_vars, build_env)

        if build_env == 'alpine':
            if cc == 'clang':
                make_vars['LLVM'] = '1'
            else:
                # Disable CROSS_COMPILE as we are going to build in a "native"
                # Alpine chroot, so there is no need for a cross compiler
                make_vars.pop('CROSS_COMPILE', None)

        return (make_vars, cc)

    @classmethod
    def _check_cc_version(cls, cc):
        if cc == 'clang':
            version = subprocess.check_output([cc, '--version'])
            m = re.match(rb'.*clang version ([0-9]+)\.', version)
            if m:
                major = int(m.group(1))
                if major >= cls._MIN_CLANG_VERSION:
                    return True
        else:
            return True

        return False

    @classmethod
    def _resolve_toolchain(cls, abi, make_vars, build_env):
        logger = cls.get_logger()
        build_env = KernelTree._resolve_build_env(build_env)

        if abi == LISA_HOST_ABI:
            toolchain = None
        else:
            try:
                toolchain = make_vars['CROSS_COMPILE']
            except KeyError:
                try:
                    toolchain = os.environ['CROSS_COMPILE']
                except KeyError:
                    if abi in ('arm64', 'aarch64'):
                        toolchain = 'aarch64-linux-gnu-'
                    elif 'arm' in abi:
                        toolchain = 'arm-linux-gnueabi-'
                    else:
                        raise KeyError('CROSS_COMPILE env var needs to be set')

                    logger.debug(f'CROSS_COMPILE env var not set, assuming "{toolchain}"')

        def test_cmd(cc):
            return [cc, *([f'--target={toolchain}'] if toolchain else []), '-x' 'c', '-c', '-', '-o', '/dev/null']

        commands = {
            # Try clang first so we can use the "musttail" return attribute
            # when possible
            **{
                cc: test_cmd(cc)
                # Try the default "clang" name first in case it's good enough
                for cc in ['clang'] + [
                    f'clang-{i}'
                    # Try the most recent ones first
                    for i in reversed(
                        # Cover for the next 10 years starting from 2021
                        range(cls._MIN_CLANG_VERSION, cls._MIN_CLANG_VERSION + 10 * 2)
                    )
                ]
            },
            'gcc': [f'{toolchain or ""}gcc', '-x' 'c', '-c', '-', '-o', '/dev/null']
        }

        cc = None

        if 'CC' in make_vars:
            cc = make_vars['CC']
            try:
                commands = {cc: commands[cc]}
            except KeyError:
                commands = {}
        # Default to clang on alpine, as it will be in a high-enough version
        # and since Alpine does not ship any cross-toolchain for GCC, this will
        # avoid having to use QEMU userspace emulation which is really slow.
        elif build_env == 'alpine':
            cc = 'clang'

        if 'LLVM' in make_vars:
            cc = cc or 'clang'
            llvm = make_vars['LLVM']
            version = llvm if llvm.startswith('-') else ''
            if cc == 'clang' and version:
                commands = {
                    cc: test_cmd(cc + version),
                }

        # Only run the check on host build env, as other build envs are
        # expected to be correctly configured.
        if build_env == 'host' and commands:
            for cc, cmd in commands.items():
                pretty_cmd = ' '.join(cmd)
                try:
                    subprocess.check_output(
                        cmd,
                        # Most basic compiler input that will not do anything.
                        input=b';',
                        stderr=subprocess.STDOUT
                    )
                except subprocess.CalledProcessError as e:
                    logger.debug(f'Checking {cc} compiler: {pretty_cmd} failed with:\n{e.output.decode()}')
                    continue
                except FileNotFoundError as e:
                    logger.debug(f'Checking {cc} compiler: {e}')
                    continue
                else:
                    if cls._check_cc_version(cc):
                        break
            else:
                raise ValueError(f'Could not find a working toolchain for CROSS_COMPILE={toolchain}')

        if cc is None:
            raise ValueError(f'Could not detect which compiler to use')

        logger.info(f'Detected CROSS_COMPILE={toolchain} and CC={cc}')

        detected = {}
        if toolchain:
            detected['CROSS_COMPILE'] = toolchain

        # For some reason Kbuild does not appreciate CC=gcc, even though
        # it's happy with CC=clang
        if cc != 'gcc':
            detected['CC'] = cc

        make_vars = {
            **detected,
            **make_vars,
        }
        return (make_vars, cc)

    @classmethod
    @SerializeViaConstructor.constructor
    def from_target(cls, target, tree_path=None, make_vars=None, cache=True, build_env=None, overlay_backend=None):
        """
        Build the tree from the given :class:`lisa.target.Target`.

        This will try multiple strategies in order to get the best kernel tree
        possible given the input:

            * Using ``/lib/modules/$(uname -r)/build`` path. This is limited by
              a number of factor since that tree is usually incomplete and can
              have symlinks pointing outside of it, making it unusable for bind
              mounts.

            * Using either a source tree or a kernel.org tarball matching
              kernel version (downloaded automatically).

            * The source tree as above, plus ``/sys/kheaders.tar.xz`` and
              ``/proc/config.gz``. This is the method that is the most likely
              to lead to a working kernel module as it allows precisely
              matching the vermagic. It will require the following configs:

                .. code-block:: sh

                    # For /sys/kheaders.tar.xz
                    CONFIG_IKHEADERS=y
                    # For /proc/config.gz
                    CONFIG_IKCONFIG=y

        :param target: Target to use.
        :type target: lisa.target.Target

        :param tree_path: If provided, a path to a kernel tree. If not given,
            other places will be tried (like /lib/modules if possible, or
            downloading a tarball from kernel.org for the matching version.)
        :type tree_path: str or None

        :param make_vars: Variables passed on ``make`` command line.
        :type make_vars: dict(str, object)

        :param cache: If ``True``, will attempt to cache intermediate steps.
        :type cache: bool

        :param build_env: See :class:`lisa._kmod.KernelTree`.
        :type build_env: str or None

        :param overlay_backend: See :class:`lisa._kmod.KernelTree`.
        :type overlay_backend: str or None
        """
        make_vars, cc = cls._process_make_vars(
            make_vars=make_vars,
            abi=target.plat_info['abi'],
            build_env=build_env,
        )
        kernel_info = target.plat_info['kernel']

        @contextlib.contextmanager
        def from_installed_headers():
            """
            Get the kernel tree from /lib/modules
            """
            if build_env == 'alpine':
                raise ValueError(f'Building from /lib/modules is not supported with the Alpine build environment as /lib/modules might not be self contained (i.e. symlinks pointing outside)')
            else:
                if isinstance(target.conn, LocalConnection):
                    # We could use this instead, except that Ubuntu does not have
                    # /proc/config.gz, so that is a bit more "reliable"
                    # target.plat_info['kernel']['config']['CONFIG_CC_VERSION_TEXT']
                    with open('/proc/version', 'r') as f:
                        proc_version = f.read()

                    # If the compiler used to build the kernel is different from the
                    # one we selected, we unfortunately cannot use the installed
                    # headers under /lib/modules, since we won't be able to re-run
                    # modules_prepare (unless we make a copy, resolving all
                    # symlinks in passing).
                    if cc in proc_version:
                        uname_r = target.execute('uname -r').strip()
                        target_path = Path('/lib', 'modules', uname_r, 'build')
                        # On a local connection, we can just directly yield the path
                        # directly rather than make a copy, as it will not be written to.
                        if target_path.is_dir():
                            # Unfortunately, we cannot use cls.from_overlays() and
                            # re-run modules_prepare, as some distro such as Ubuntu
                            # create build folders full of relative symlinks
                            # pointing outside of it, which renders it unusable in
                            # an overlay. Without the overlay, we cannot modify
                            # anything since the folder is owned by an apt package.
                            yield dict(
                                path=target_path,
                                # Since we basically know it's the distro kernel,
                                # we can cache the result
                                checksum=hashlib.sha256(
                                    f'distro-kernel-{uname_r}'.encode()
                                ).hexdigest()
                            )
                        else:
                            raise ValueError(f'{target_path} is not a folder')
                    else:
                        raise ValueError(f'The chosen compiler ({cc}) is different from the one used to build the kernel ({proc_version}), /lib/modules/ tree will not be used')
                else:
                    raise ValueError(f'Building from /lib/modules/.../build/ is only supported for local targets')

        @contextlib.contextmanager
        def from_sysfs_headers():
            """
            From /sys/kheaders.tar.xz
            """
            version = kernel_info['version']
            config = kernel_info['config']
            if not (
                config.get('CONFIG_IKHEADERS') == KernelConfigTristate.YES and
                config.get('CONFIG_IKCONFIG_PROC') == KernelConfigTristate.YES
            ):
                raise ValueError('Needs CONFIG_IKHEADERS=y and CONFIG_IKCONFIG_PROC=y')
            else:
                with tempfile.TemporaryDirectory() as temp:
                    temp = Path(temp)
                    target.cached_pull('/proc/config.gz', str(temp))
                    target.cached_pull('/sys/kernel/kheaders.tar.xz', str(temp), via_temp=True)

                    with cls.from_overlays(
                        version=version,
                        overlays={
                            FileOverlay.from_path(temp / 'config.gz', decompress=True): '.config',
                            TarOverlay.from_path(temp / 'kheaders.tar.xz'): '.',
                        },
                        make_vars=make_vars,
                        cache=cache,
                        tree_path=tree_path,
                        build_env=build_env,
                        overlay_backend=overlay_backend,
                    ) as tree:
                        yield tree._to_spec()

        @contextlib.contextmanager
        def from_user_tree():
            """
            Purely from the tree passed by the user.
            """
            if tree_path is None:
                raise ValueError('Use tree_path != None to build from a user-provided tree')
            else:
                # We still need to run make modules_prepare on the provided
                # tree
                with cls.from_overlays(
                    tree_path=tree_path,
                    version=kernel_info['version'],
                    cache=cache,
                    make_vars=make_vars,
                    build_env=build_env,
                    overlay_backend=overlay_backend,
                ) as tree:
                    yield tree._to_spec()

        @contextlib.contextmanager
        def try_loaders(loaders):
            logger = cls.get_logger()
            exceps = []
            inner_excep = None
            for loader in loaders:
                logger.debug(f'Trying to load kernel tree using loader {loader.__name__} ...')
                try:
                    with loader() as spec:
                        try:
                            yield spec
                            break
                        except BaseException as e:
                            inner_excep = e
                            raise
                except Exception as e:
                    if inner_excep is e:
                        raise
                    else:
                        logger.debug(f'Failed to load kernel tree using loader {loader.__name__}: {e.__class__.__name__}: {e}')
                        exceps.append((loader, e))
                else:
                    logger.debug(f'Loaded kernel tree using loader {loader.__name__}')

            else:
                excep_str = "\n".join(
                    f"{loader.__name__}: {e.__class__.__name__}: {e}"
                    for loader, e in exceps
                )
                raise ValueError(f'Could not load kernel trees:\n{excep_str}')

        # Try these loaders in the given order, until one succeeds
        loaders = [from_installed_headers, from_sysfs_headers, from_user_tree]

        return cls(
            path_cm=functools.partial(try_loaders, loaders),
            make_vars=make_vars,
            build_env=build_env,
            overlay_backend=overlay_backend,
        )

    @classmethod
    @SerializeViaConstructor.constructor
    def from_path(cls, path, make_vars=None, cache=True, build_env=None):
        """
        Build a tree from the given ``path`` to sources.
        """
        return cls.from_overlays(
            tree_path=path,
            make_vars=make_vars,
            cache=cache,
            build_env=build_env,
        )

    @classmethod
    @SerializeViaConstructor.constructor
    def from_overlays(cls, version=None, tree_path=None, overlays=None, make_vars=None, cache=True, build_env=None, overlay_backend=None):
        """
        Build a tree from the given overlays, to be applied on a source tree.

        :param version: Version of the kernel to be used.
        :type version: devlib.target.KernelVersion or str

        :param overlays: List of overlays to apply on the tree.
        :type overlays: list(OverlayResource)
        """
        logger = cls.get_logger()
        overlays = overlays or {}
        make_vars, cc = cls._process_make_vars(
            make_vars=make_vars,
            build_env=build_env,
        )
        build_env = KernelTree._resolve_build_env(build_env)
        overlay_backend = KernelTree._resolve_overlay_backend(overlay_backend)

        if tree_path:
            try:
                config = Path(tree_path, '.config').read_bytes()
            except FileNotFoundError:
                restore_config = lambda _: None
            else:
                restore_config = lambda path: path.write_bytes(config)
        else:
            restore_config = lambda _: None

        def copy_filter(src, dst, remove_obj=False):
            return not (
                (remove_obj and (src.suffix == '.o')) or
                any(
                    # Skip some folders that are useless to build a kernel
                    # module.
                    path.name == '.git'
                    for path in src.parents
                )
            )

        def apply_overlays(path):
            # Ensure .config is present if it was at the beginning, so that it
            # survives make mrproper in _prepare_tree()
            restore_config(path / '.config')

            for overlay, dst in overlays.items():
                logger.debug(f'Unpacking overlay {overlay} -> {dst}')
                overlay.write_to(os.path.join(path, dst))

        def prepare_overlay(path):
            cls._prepare_tree(
                path,
                make_vars=make_vars,
                build_env=build_env,
                apply_overlays=functools.partial(apply_overlays, path),
                overlay_backend=overlay_backend,
            )

        @contextlib.contextmanager
        def overlay_cm(args):
            base_path, tree_key = args
            base_path = Path(base_path).resolve()

            if cache and tree_key is not None:
                # Compute a unique token for the overlay. It includes:
                # * The hash of all overlays resources. It should be
                #   relatively inexpensive to compute, as most overlays are
                #   pretty small.
                # * The key that comes with the tree passed as base_path, if it
                #   comes from a reliably read-only source.
                # * The build environment
                # * All the variables passed to "make". This is very important
                #   as things such as a toolchain change can make a kernel tree
                #   unsuitable for compiling a module.
                key = (
                    sorted(
                        overlay._get_checksum()
                        for overlay, dst in overlays.items()
                    ) + [
                        str(tree_key),
                        str(build_env),
                    ] + [
                        # We need to take checksum the make variables
                        # as well, as it can influence the kernel tree
                        # a great deal (e.g. changing toolchain)
                        f'{k}={v}'
                        for k, v in sorted((make_vars or {}).items())
                    ]
                )

                def populate(key, path):
                    # Prepare the overlay separately from the final
                    # overlayfs mount point, so that we never end up
                    # sharing the same upper between 2 mounts, which is not
                    # allowed by the kernel.
                    with _overlay_folders(
                        lowers=[base_path],
                        upper=path,
                        backend=overlay_backend,
                        copy_filter=functools.partial(copy_filter, remove_obj=True)
                    ) as path:
                        prepare_overlay(path)

                dir_cache = DirCache(
                    category='kernels_overlays',
                    populate=populate,
                )
                cache_path = dir_cache.get_entry(key)
                with _overlay_folders([base_path, cache_path], backend=overlay_backend, copy_filter=copy_filter) as path:
                    yield dict(
                        path=path,
                        checksum=dir_cache.get_key_token(key),
                    )
            else:
                with _overlay_folders([base_path], backend=overlay_backend, copy_filter=copy_filter) as path:
                    prepare_overlay(path)
                    yield dict(
                        path=path,
                        checksum=None,
                    )

        if not (version is None or isinstance(version, KernelVersion)):
            version = KernelVersion(version)

        @contextlib.contextmanager
        def tree_cm():
            if tree_path:
                logger.debug(f'Using provided kernel tree at: {tree_path}')
                try:
                    repo_root = git.find_root(tree_path)
                    sha1 = git.get_sha1(tree_path)
                    patch = git.get_uncommited_patch(tree_path)
                except (FileNotFoundError, subprocess.CalledProcessError):
                    key = None
                else:
                    if repo_root.resolve() == Path(tree_path).resolve():
                        patch_sha1 = hashlib.sha1(patch.encode()).hexdigest()
                        key = f'{sha1}-{patch_sha1}'
                    else:
                        key = None
                yield (tree_path, key)
            elif version is None:
                raise ValueError('Kernel version is required in order to download the kernel sources')
            elif cache:
                dir_cache = DirCache(
                    category='kernels',
                    populate=lambda url, path: cls._make_tree(version, path),
                )

                url = cls._get_url(version)
                # Assume that the URL will always provide the same tarball
                yield (
                    dir_cache.get_entry([url]),
                    url,
                )
            else:
                with tempfile.TemporaryDirectory() as path:
                    yield (
                        cls._make_tree(version, path),
                        version,
                    )

        cm = chain_cm(overlay_cm, tree_cm)
        return cls(
            path_cm=cm,
            make_vars=make_vars,
            build_env=build_env,
        )

    @classmethod
    def _make_tree(cls, version, path):
        response = cls._open_url(version)
        filename = _url_path(response.url).name
        tar_path = os.path.join(path, filename)

        extract_folder = os.path.join(path, 'extract')

        # Unfortunately we can't feed url_f directly to tarfile as it needs to
        # seek()
        try:
            with response as url_f, open(tar_path, 'wb') as tar_f:
                shutil.copyfileobj(url_f, tar_f)

            with tarfile.open(tar_path) as tar:
                # Account for a top-level folder in the archive
                prefix = os.path.commonpath(
                    member.path
                    for member in tar.getmembers()
                )
                tar.extractall(extract_folder)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(extract_folder, ignore_errors=True)
            raise
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tar_path)

        return os.path.join(extract_folder, prefix)



class KmodSrc(Loggable):
    """
    Sources of a kernel module.

    :param src: Mapping of source path to file content.
    :type src: dict(str, str)

    :param name: If provided, sets the name of the module. If ``None``, a name
        will be created using a checksum of the sources.
    :type name: str or None
    """
    def __init__(self, src, name=None):
        def encode(x):
            if isinstance(x, str):
                return x.encode('utf-8')
            else:
                return x

        self.src = {
            name: encode(content)
            for name, content in src.items()
        }
        self._mod_name = name

    @property
    def c_files(self):
        return {
            name: content
            for name, content in self.src.items()
            if name.endswith('.c')
        }

    @property
    @memoized
    def checksum(self):
        """
        Checksum of the module's sources.
        """
        m = hashlib.sha256()
        for name, content in sorted(self.src.items()):
            m.update(name.encode('utf-8'))
            m.update(content)
        return m.hexdigest()

    @property
    @memoized
    def mod_name(self):
        """
        Name of the module.
        """
        if self._mod_name:
            return self._mod_name
        else:
            # Kernel macro MODULE_NAME_LEN
            max_size = 64 - 8 - 1
            return f'lisa-{self.checksum}'[:max_size]

    @property
    @memoized
    def makefile(self):
        try:
            return self.src['Kbuild']
        except KeyError:
            try:
                return self.src['Makefile']
            except KeyError:
                name = self.mod_name
                return '\n'.join((
                    f'obj-m := {name}.o',
                    f'{name}-y := ' + ' '.join(
                        f'{Path(filename).stem}.o'
                        for filename in sorted(self.c_files.keys())
                    )
                )).encode('utf-8')

    def compile(self, kernel_tree, make_vars=None):
        """
        Compile the module and returns the ``bytestring`` content of the
        ``.ko`` file.

        :param kernel_tree: Kernel tree to build the module against.
        :type kernel_tree: KernelTree

        :param make_vars: Variables passed on ``make`` command line. This can
            be used for variables only impacting the module, otherwise it's
            better to set them when creating the ``kernel_tree``.
        :type make_vars: dict(str, object) or None
        """
        make_vars = dict(make_vars or {})
        overlay_backend = kernel_tree.overlay_backend
        tree_path = Path(kernel_tree.path)
        # "inherit" the build env from the KernelTree as we must use the same
        # environment as what was used for "make modules_prepare"
        build_env = kernel_tree.build_env
        bind_paths = {tree_path: tree_path}
        logger = self.logger

        def populate_mod(path):
            mod_path = Path(path)

            src = {
                **self.src,
                'Kbuild': self.makefile,
            }

            for name, content in src.items():
                file_path = mod_path / name
                file_path.parent.mkdir(exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(content)

        def make_cmd(tree_path, mod_path, make_vars):
            make_vars = make_vars.copy()
            make_vars.update(
                M=mod_path,
                LISA_KMOD_NAME=self.mod_name,
            )
            make_vars = [
                f'{name}={value}'
                for name, value in sorted(make_vars.items())
                if value is not None
            ]
            cmd = ['make', '-C', tree_path, *make_vars, 'modules']
            return cmd

        def find_mod_file(path):
            filenames = glob.glob(str(path.resolve() / '*.ko'))

            if not filenames:
                raise FileNotFoundError(f'Could not find .ko file in {path}')
            elif len(filenames) > 1:
                raise ValueError(f'Found more than one .ko file in {path}: {filenames}')
            else:
                return filenames[0]

        if build_env == 'alpine':
            @contextlib.contextmanager
            def cmd_cm():
                with _make_chroot(bind_paths=bind_paths, make_vars=make_vars, overlay_backend=overlay_backend) as chroot:
                    # Do not use a CM here to avoid choking on permission
                    # issues. Since the chroot itself will be entirely
                    # removed it's not a problem.
                    mod_path = Path(tempfile.mkdtemp(dir=chroot / 'tmp'))
                    cmd = make_cmd(
                        tree_path=tree_path,
                        mod_path=f'/{mod_path.relative_to(chroot)}',
                        make_vars=make_vars,
                    )
                    yield (mod_path, _make_chroot_cmd(chroot, cmd), {})
        else:
            @contextlib.contextmanager
            def cmd_cm():
                with tempfile.TemporaryDirectory() as mod_path:
                    cmd = make_cmd(
                        tree_path=tree_path,
                        mod_path=mod_path,
                        make_vars=make_vars,
                    )
                    yield (mod_path, cmd, {'PATH': HOST_PATH})

        with cmd_cm() as (mod_path, cmd, env):
            mod_path = Path(mod_path)
            populate_mod(mod_path)

            logger.info(f'Compiling kernel module {self.mod_name}')
            _subprocess_log(cmd, logger=logger, level=logging.DEBUG, extra_env=env)

            mod_file = find_mod_file(mod_path)
            with open(mod_file, 'rb') as f:
                return f.read()

    @classmethod
    def from_path(cls, path, extra=None, **kwargs):
        """
        Build an instance from the path to the sources.

        :param extra: Extra sources to use, same as ``src`` parameter
            :class:`lisa._kmod.KmodSrc`.
        :type extra: dict(str, str)
        """
        def get_files(root, dirs, files):
            for f in files:
                yield (Path(root) / f).resolve()

        path = Path(path).resolve()
        src = {
            str(f.relative_to(path)): f.read_bytes()
            for files in itertools.starmap(get_files, os.walk(path))
            for f in files
        }
        src.update(extra or {})

        return cls(src=src, **kwargs)


class CannotLoadModuleError(Exception):
    """
    Raised when a kernel module cannot be loaded (or will not be loaded because
    of nasty side effects).
    """


class DynamicKmod(Loggable):
    """
    Dynamic kernel module that can be compiled on the go by LISA.

    :param target: Target that will be used to load the module.
    :type target: lisa.target.Target

    :param src: Sources of the module.
    :type src: lisa._kmod.KmodSrc

    :param kernel_tree: Kernel source tree to use to build the module against.
    :type kernel_tree: lisa._kmod.KernelTree
    """
    def __init__(self, target, src, kernel_tree=None):
        self.src = src
        self.target = target

        if not isinstance(kernel_tree, KernelTree):
            kernel_tree = KernelTree.from_target(
                target=self.target,
                tree_path=kernel_tree,
            )

        self._kernel_tree = kernel_tree

    @classmethod
    def from_target(cls, target, **kwargs):
        """
        Build a module from the given target. Use this constructor on
        subclasses rather than making assumptions on the signature of the
        class.

        :Variable keyword arguments: Forwarded to ``__init__``.
        """
        return cls(target=target, **kwargs)

    @property
    @memoized
    def kernel_tree(self):
        tree = self._kernel_tree
        arch = _any_abi_to_kernel_arch(
            self.target.plat_info['abi']
        )
        tree_arch = tree.make_vars['ARCH']
        if tree_arch != arch:
            raise ValueError(f'The kernel tree ({tree_arch}) was not prepared for the same architecture as the target ({arch}). Please set ARCH={arch} make variable.')
        else:
            return tree

    @property
    def _compile_needs_root(self):
        tree = self.kernel_tree
        return (
            tree.build_env != 'host' or
            tree.overlay_backend == 'overlayfs'
        )

    # Dummy memoized wrapper. The only reason we need one is that _do_compile()
    # needs to be pickleable to be sent to a multiprocessing Process, so it
    # cannot be overriden by a wrapper
    @memoized
    def _compile(self):
        compile_ = self._do_compile.__func__
        if self._compile_needs_root:
            compile_ = ensure_root(compile_, inline=True)

        bin_, spec = compile_(self)
        # Get back KernelTree._to_spec() and update the KernelTree we have in
        # this process with it to remember the checksum, in case ensure_root()
        # spawned a new process. This is then used by Target.get_kmod() that
        # will reinject the known spec when creating new modules from the
        # default KernelTree
        self.kernel_tree._update_spec(spec)
        return bin_

    def _do_compile(self):
        kernel_tree = self.kernel_tree
        src = self.src

        def get_key(kernel_tree):
            kernel_checksum = kernel_tree.checksum
            if kernel_checksum is None:
                raise ValueError('Kernel tree has no checksum')
            else:
                var_tokens = [
                    f'{k}={v}'
                    for k, v in sorted(kernel_tree.make_vars.items())
                ]
                # Cache the compilation based on:
                # * the kernel tree
                # * the make variables
                # * the module name
                return (kernel_checksum, kernel_tree.build_env, src.checksum, *var_tokens)

        def get_bin(kernel_tree):
            return src.compile(
                kernel_tree=kernel_tree,
                make_vars=kernel_tree.make_vars,
            )

        def lookup_cache(kernel_tree, key, enter_cm=False):
            cm = kernel_tree if enter_cm else nullcontext(kernel_tree)

            def populate(key, path):
                with cm as kernel_tree:
                    with open(path / 'mod.ko', 'wb') as f:
                        f.write(get_bin(kernel_tree))

            dir_cache = DirCache(
                category='kernel_modules',
                populate=populate,
            )
            cache_path = dir_cache.get_entry(key)
            with open(cache_path / 'mod.ko', 'rb') as f:
                return f.read()

        # First try on the "bare" kernel tree, i.e. before calling __enter__().
        # If this happens to have enough information to hit the cache, we just
        # avoided a possibly costly setup of compilation environment
        try:
            key = get_key(kernel_tree)
        except ValueError:
            with kernel_tree as kernel_tree:
                if kernel_tree.checksum is None:
                    # Only cache the module if the kernel tree has a defined
                    # checksum, which is not always the case when it's not
                    # coming from a controlled source that is guaranteed to be
                    # immutable.
                    bin_ = get_bin(kernel_tree)
                else:
                    key = get_key(kernel_tree)
                    bin_ = lookup_cache(kernel_tree, key)
        else:
            bin_ = lookup_cache(kernel_tree, key, enter_cm=True)

        return (bin_, kernel_tree._to_spec())

    def install(self, kmod_params=None):
        """
        Install and load the module on the target.

        :param kmod_params: Parameters to pass to the module via ``insmod``.
            Non-string iterable values will be turned into a comma-separated
            string following the ``module_param_array()`` kernel API syntax.
        :type kmod_params: dict(str, object) or None
        """
        def target_mktemp():
            return target.execute(
                f'mktemp -p {quote(target.working_directory)}'
            ).strip()

        def make_str(x):
            if isinstance(x, str):
                return x
            elif isinstance(x, collections.abc.Iterable):
                return ','.join(map(make_str, x))
            else:
                return str(x)

        target = self.target
        content = self._compile()

        kmod_params = kmod_params or {}
        params = ' '.join(
            f'{quote(k)}={quote(make_str(v))}'
            for k, v in sorted(
                kmod_params.items(),
                key=itemgetter(0),
            )
        )

        with tempfile.NamedTemporaryFile('wb', suffix='.ko') as f:
            f.write(content)
            f.flush()

            temp_ko = target_mktemp()
            try:
                target.push(f.name, temp_ko)
                target.execute(f'insmod {quote(temp_ko)} {params}', as_root=True)
            finally:
                target.remove(temp_ko)

    def uninstall(self):
        """
        Unload the module from the target.
        """
        self.target.execute(f'rmmod {quote(self.src.mod_name)}')

    @contextlib.contextmanager
    def run(self, **kwargs):
        """
        Context manager used to run the module by loading it then unloading it.

        :Variable keyword arguments: Forwarded to :meth:`install`.
        """
        try:
            self.uninstall()
        except Exception:
            pass

        x = self.install(**kwargs)
        try:
            yield x
        finally:
            self.uninstall()


class FtraceDynamicKmod(DynamicKmod):
    """
    Dynamic module providing some custom ftrace events.
    """

    def install(self, *args, **kwargs):
        config = self.target.plat_info['kernel']['config']

        # Check for CFI as registering tracepoint probes apparently results in
        # violations
        if config.get('CFI_CLANG', False) and not config.get('CFI_PERMISSIVE', False):
            raise CannotLoadModuleError('CFI (Control-flow integrity) is enabled and loading the module will crash the kernel. Setting CONFIG_CFI_PERMISSIVE=y will turn the crash into a warning')
        else:
            return super().install(*args, **kwargs)

    def _get_symbols(self, section=None):
        content = self._compile()
        elf = ELFFile(BytesIO(content))

        if section:
            section_idx = {
                s.name: idx
                for idx, s in enumerate(elf.iter_sections())
            }

            idx = section_idx[section]
            predicate = lambda s: s.entry['st_shndx'] == idx
        else:
            predicate = lambda s: True

        symtab = elf.get_section_by_name('.symtab')
        return sorted(
            s.name
            for s in symtab.iter_symbols()
            if predicate(s)
        )

    @property
    @memoized
    def defined_events(self):
        """
        Ftrace events defined in that module.
        """
        def parse(name):
            return re.match(r'__event_(.*)', name)

        return sorted(set(
            m.group(1)
            for m in map(
                parse,
                self._get_symbols('_ftrace_events')
            )
            if m
        ))


class LISAFtraceDynamicKmod(FtraceDynamicKmod):
    """
    Module providing ftrace events used in various places by :mod:`lisa`.

    The kernel must be compiled with the following options in order for the
    module to be created successfully:

    .. code-block:: sh

        CONFIG_DEBUG_INFO=y
        CONFIG_DEBUG_INFO_BTF=y
        CONFIG_DEBUG_INFO_REDUCED=n
    """

    @classmethod
    def from_target(cls, target, **kwargs):
        path = Path(ASSETS_PATH) / 'kmodules' / 'sched_tp'
        btf_path = '/sys/kernel/btf/vmlinux'

        with tempfile.NamedTemporaryFile() as f:
            try:
                target.cached_pull(btf_path, f.name, via_temp=True)
            except FileNotFoundError:
                raise FileNotFoundError(f'Could not find {btf_path} on the target. Ensure you compiled your kernel using CONFIG_DEBUG_INFO=y CONFIG_DEBUG_INFO_BTF=y CONFIG_DEBUG_INFO_REDUCED=n')

            with open(f.name, 'rb') as f:
                btf = f.read()

        extra = {
            'vmlinux': btf
        }
        src = KmodSrc.from_path(path, extra=extra, name='lisa')
        return cls(
            target=target,
            src=src,
            **kwargs,
        )

    @classmethod
    def _event_features(cls, events):
        return set(f'event__{event}' for event in events)

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
