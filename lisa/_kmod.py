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
import itertools
import logging
import datetime
import pwd
import glob
import collections
import collections.abc
import hashlib
import errno
from operator import itemgetter
from shlex import quote
from io import BytesIO
from collections.abc import Mapping
import typing
import fnmatch
import sys
from enum import IntEnum
import traceback
import uuid
import textwrap

from elftools.elf.elffile import ELFFile

from devlib.target import AndroidTarget, KernelVersion, TypedKernelConfig, KernelConfigTristate
from devlib.host import LocalConnection
from devlib.exception import TargetStableError

from lisa.utils import nullcontext, Loggable, LISA_CACHE_HOME, checksum, DirCache, chain_cm, memoized, LISA_HOST_ABI, subprocess_log, SerializeViaConstructor, destroyablecontextmanager, ContextManagerExit, ignore_exceps, get_nested_key, is_link_dead, deduplicate, subprocess_detailed_excep
from lisa._assets import ASSETS_PATH, HOST_PATH, ABI_BINARIES_FOLDER
from lisa._unshare import ensure_root
import lisa._git as git
from lisa.conf import SimpleMultiSrcConf, TopLevelKeyDesc, LevelKeyDesc, KeyDesc, VariadicLevelKeyDesc
from lisa._kallsyms import parse_kallsyms

_KERNEL_BINUTILS = (
    'ld',
    'ar',
    'nm',
    'strip',
    'objcopy',
    'objdump',
    'readelf',
)

def _tar_extractall(f, *args, **kwargs):
    # Avoid DeprecationWarning, see:
    # https://docs.python.org/3/library/tarfile.html#extraction-filters
    if sys.version_info[:2] >= (3, 12):
        kwargs['filter'] = 'tar'
    return f.extractall(*args, **kwargs)


def _make_vars_cc(make_vars, default=None):
    try:
        cc = make_vars['CC']
    except KeyError:
        if default is None:
            raise
        else:
            cc = default
    return Path(cc)


class KmodVersionError(Exception):
    """
    Raised when the kernel module is not found with the expected version.
    """
    pass


_ALPINE_DEFAULT_VERSION = '3.20.3'
_ALPINE_ROOTFS_URL = 'https://dl-cdn.alpinelinux.org/alpine/v{minor}/releases/{arch}/alpine-minirootfs-{version}-{arch}.tar.gz'
_ALPINE_PACKAGE_INFO_URL = 'https://pkgs.alpinelinux.org/package/v{version}/{repo}/{arch}/{package}'


def _get_alpine_clang_packages(cc):
    llvm_version = _clang_version_static(cc) or ''
    return [
        f'clang{llvm_version}',
        f'llvm{llvm_version}',
        # "lld" packaging is a bit strange, any versioned lld (e.g. "lld15")
        # conflicts with the generic "lld" package. On top of that, there is
        # only one versionned package ("lld15") as of Alpine v3.18.
        f'lld'
    ]


@functools.lru_cache(maxsize=256, typed=True)
def _find_alpine_cc_packages(version, abi, cc, cross_compile):
    logger = logging.getLogger(f'{__name__}.alpine_chroot.packages')

    if 'gcc' in cc and cross_compile:
        cross_compile = cross_compile.strip('-')
        packages = [f'gcc-{cross_compile}']
    elif 'clang' in cc:
        packages = _get_alpine_clang_packages(cc)
    else:
        packages = [cc]

    def check(repo, package):
        url = _ALPINE_PACKAGE_INFO_URL.format(
            version='.'.join(map(str, version[:2])),
            repo=repo,
            arch=_abi_to_alpine_arch(abi),
            package=package,
        )
        logger.debug(f'Checking Alpine package URL: {url}')
        return not is_link_dead(url)

    ok = all(
        any(
            check(repo, package)
            for repo in ('main', 'community')
        )
        for package in packages
    )

    if ok:
        return packages
    else:
        raise ValueError(f'Could not find Alpine linux packages: {", ".join(packages)}')


def _abi_to_kernel_arch(abi):
    """
    Convert a devlib ABI into a valid ARCH= for the kernel
    """
    return {
        'armeabi': 'arm',
    }.get(abi, abi)


def _kernel_arch_to_abi(arch):
    """
    Convert a kernel arch to a devlib ABI
    """
    if arch == 'arm64':
        return 'arm64'
    elif 'arm' in arch:
        return 'armeabi'
    else:
        return arch

def _abi_to_alpine_arch(abi):
    return {
        'arm64': 'aarch64',
        'armeabi': 'armv7',
    }.get(abi, abi)


def _url_path(url):
    return PurePosixPath(
        urllib.parse.unquote(
            urllib.parse.urlparse(url).path
        )
    )


def _filter_env(env):
    allowed = {
        'PATH',
        'USER',
        'TERM',
        'TMPDIR',
        # Allow any env var set inside our code that is not already in
        # os.environ
        *(
            set(env.keys()) - set(os.environ.keys())
        )
    }
    return {
        k: v
        for k, v in env.items()
        if k in allowed
    }


def _subprocess_log(*args, env=None, **kwargs):
    env = env or os.environ
    env = _filter_env(env)

    with subprocess_detailed_excep():
        return subprocess_log(*args, **kwargs, env=env)


def _kbuild_make_cmd(path, targets, cc, make_vars):
    make_vars = make_vars or {}

    remove = []
    try:
        _cc = _make_vars_cc(make_vars)
    except KeyError:
        pass
    else:
        # For some reason Kbuild does not appreciate CC=gcc, even though it's
        # happy with CC=clang
        if str(_cc) == 'gcc':
            remove.append('CC')
        # If we have a path to a specific GCC, we don't want to pass
        # CROSS_COMPILE as we probably have someone pointing at a specific
        # toolchain folder. In that case, passing CROSS_COMPILE would break the
        # path to GCC
        elif 'gcc' in _cc.name:
            remove.append('CROSS_COMPILE')

    formatted_vars = [
        f'{name}={val}'
        for name, val in sorted(make_vars.items())
        if (
            name not in remove and
            val is not None
        )
    ]

    nr_cpus = os.cpu_count()

    cmd = ['make', f'-j{nr_cpus}', '-C', path, '--', *formatted_vars, *targets]

    var_cc = make_vars.get('CC', cc)
    if str(var_cc) != str(cc):
        pretty_cmd = ' '.join(map(quote, map(str, cmd)))
        raise ValueError(f'The kernel build env was prepared using CC={cc} so the make command cannot be ran with CC={var_cc}: {pretty_cmd}')

    return cmd


def _clang_version_static(cc):
    try:
        _, version = cc.split('-', 1)
    except ValueError:
        # apk understands "clang" even if there is no clang package
        version = None
    else:
        version = int(version)

    return version


def _clang_version(cc, env):
    version = subprocess.check_output([cc, '--version'], env=env)
    m = re.match(rb'.*clang version ([0-9]+)\.', version)
    if m:
        major = int(m.group(1))
        return (major,)
    else:
        raise ValueError(f'Could not determine version of {cc}')


def _install_rust(rust_spec, run_cmd):
    rust_version = rust_spec['version']
    components = sorted(rust_spec.get('components', []))
    crates = rust_spec.get('crates', {})

    rustup_home = str(rust_spec['rustup_home'])
    cargo_home = str(rust_spec['cargo_home'])
    cargo_bin = str(Path(cargo_home) / 'bin')

    def run(cmd):
        cmd = f'export PATH={cargo_bin}:$PATH RUSTC_BOOTSTRAP=1 RUSTUP_INIT_SKIP_PATH_CHECK=yes RUSTUP_HOME={quote(rustup_home)} CARGO_HOME={quote(cargo_home)} && {cmd}'
        run_cmd(['sh', '-c', cmd])

    # Install rustup
    with urllib.request.urlopen('https://sh.rustup.rs') as response:
        script = response.read()
    script = script.decode('utf-8')
    run(f'sh -c {quote(script)} -- -y --no-modify-path --default-toolchain none')

    # Install rust toolchain
    components = ' '.join(
        f'--component={quote(compo)}'
        for compo in components
    )
    run(f"rustup toolchain install {quote(rust_version)} --profile minimal {components}")

    for crate, crate_spec in sorted(crates.items()):
        run(f"cargo +{rust_version} install --locked --version {quote(crate_spec['version'])} {quote(crate)}")


def _resolve_alpine_version(version):
    version = version or _ALPINE_DEFAULT_VERSION

    # Ensure we have a full version number with 3 components
    version = version.lstrip('v').split('.')
    version = tuple(map(int, version + ['0' for _ in range(3 - len(version))]))
    return version


@destroyablecontextmanager
def _make_build_chroot(cc, cross_compile, abi, bind_paths=None, version=None, overlay_backend=None, packages=None, rust_spec=None):
    """
    Create a chroot folder ready to be used to build a kernel.
    """
    logger = logging.getLogger(f'{__name__}.alpine_chroot')

    if (version, packages) != (None, None) and None in (version, packages):
        raise ValueError('Both version and packages need to be set or none of them')
    else:
        version = _resolve_alpine_version(version)

        def is_clang(cc):
            return cc.startswith('clang')

        def default_packages(cc):
            maybe_qemu = False

            # Default packages needed to compile a linux kernel module
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
                'py3-pip',
                'perl',
                'pahole',
                'git',
            ]

            if is_clang(cc):
                packages.extend([
                    # Add version-less packages as well, so that userspace tools
                    # relying on "clang" when LLVM=1 is passed can work.
                    'llvm',
                    'clang',
                    'lld',
                ])

            return packages

        def resolve_toolchain_packages(cc, cross_compile):
            maybe_qemu = False
            try:
                toolchain_packages = _find_alpine_cc_packages(
                    version=version,
                    abi=abi,
                    cc=cc,
                    cross_compile=cross_compile,
                )
            except ValueError:
                # We could not find the cross compilation toolchain, so
                # fallback on the non-cross toolchain and use QEMU
                toolchain_packages = [cc]
                # clang is always a cross compilation, toolchain so we
                # would not need QEMU for that
                maybe_qemu = not is_clang(cc)

            return (toolchain_packages, maybe_qemu)

        if packages is None:
            toolchain_packages, maybe_qemu = resolve_toolchain_packages(cc, cross_compile)
            packages = default_packages(cc) + toolchain_packages
        else:
            _, maybe_qemu = resolve_toolchain_packages(cc, cross_compile)

        use_qemu = (
            maybe_qemu and
            # Since clang binaries support cross compilation without issues,
            # there is no need to use QEMU that will slow everything down.
            abi != LISA_HOST_ABI
        )

        chroot_abi = abi if use_qemu else LISA_HOST_ABI

        bind_paths = {
            **dict(bind_paths or {}),
            ABI_BINARIES_FOLDER[chroot_abi]: '/usr/local/bin'
        }

        with _make_alpine_chroot(
            version=version,
            abi=chroot_abi,
            packages=packages,
            bind_paths=bind_paths,
            overlay_backend=overlay_backend,
            rust_spec=rust_spec,
        ) as chroot:
            try:
                yield chroot
            except ContextManagerExit:
                pass


def default_llvm_tool_name(tool, llvm):
    if tool == 'clang':
        return f'clang{llvm}'
    elif tool == 'ld':
        return f'ld.lld{llvm}'
    else:
        return f'llvm-{tool}{llvm}'


def alpine_llvm_tool_name(tool, llvm):
    """
    Alpine has a different naming convention than most other distros.

    See:
    https://pkgs.alpinelinux.org/contents?name=llvm15&repo=main&branch=edge&arch=aarch64
    """
    if tool == 'clang':
        return f'clang{llvm}'
    elif tool == 'ld':
        return f'ld.lld{llvm}'
    else:
        version = llvm.strip('-') if llvm else ''
        return f'llvm{version}-{tool}'


def reroot(new_root, path):
    path = Path(path)
    assert path.is_absolute()
    path = path.relative_to('/')
    return new_root / path


@destroyablecontextmanager
def _make_alpine_chroot(version, packages=None, abi=None, bind_paths=None, overlay_backend='overlayfs', rust_spec=None):
    logger = logging.getLogger(f'{__name__}.alpine_chroot')

    def mount_binds(chroot, bind_paths, mount=True):
        for src, dst in bind_paths.items():
            src = Path(src)
            dst = reroot(chroot, dst)
            # This will be unmounted by the destroy script
            if mount:
                for p in (src, dst):
                    p.mkdir(parents=True, exist_ok=True)
                cmd = ['mount', '--rbind', '--', src, dst]
            else:
                cmd = ['umount', '-nl', '--', dst]
            _subprocess_log(cmd, logger=logger, level=logging.DEBUG)

    def run_cmd_host(cmd):
        _subprocess_log(cmd, logger=logger, level=logging.DEBUG)

    def run_cmd_chroot(chroot_path, cmd):
        cmd = _make_build_chroot_cmd(chroot_path, cmd)
        run_cmd_host(cmd)

    def populate(key, path, stage='init'):
        version, alpine_arch, packages = key
        path = path.resolve()
        run_cmd = lambda cmd: run_cmd_chroot(path, cmd)

        def fixup_llvm_tools(root, packages):
            """
            Alpine has packages with version names for LLVM and clang (e.g.
            llvm17) but the actual binaries in that package may or may not have
            the version number in them.

            They only have it if the package is not the default version for the
            tool. E.g. Alpine 3.20 uses clang 17 as default for clang, so the
            clang17 will ship a command named "clang" instead of "clang-17".
            Packages for the non-default versions will ship binaries with the
            version name in them, e.g. "clang-18" binary in the "clang18"
            package.
            """
            regex = re.compile(r'(?P<tool>[^0-9]*)(?P<version>[0-9]*)')
            packages = [
                package
                for package in packages
                if any(
                    x in package
                    for x in ('llvm', 'clang')
                )
            ]
            packages = [
                (tool, version)
                for package in packages
                if (
                    (m := regex.match(package)) and
                    (version := m.group('version')) and
                    (tool := m.group('tool'))
                )
            ]

            def expand_tool(tool):
                if tool == 'llvm':
                    return _KERNEL_BINUTILS
                else:
                    return (tool,)

            commands = [
                (
                    alpine_llvm_tool_name(tool, None),
                    alpine_llvm_tool_name(tool, f'-{version}')
                )
                for _tool, version in packages
                for tool in expand_tool(_tool)
            ]

            for unversioned_cmd, versioned_cmd in commands:
                try:
                    run_cmd(['which', versioned_cmd])
                except subprocess.CalledProcessError:
                    logger.debug(f'Creating "{versioned_cmd}" shim for "{unversioned_cmd}" since this version of Alpine ships an unversioned binary name for that tool/version combination')
                    # We just write in /usr/bin instead of e.g. /usr/local/bin
                    # since /usr/local/bin is already bind-mounted to a folder
                    # in LISA with the binaries we ship for that arch.
                    #
                    # Also, we are done installing packages so it's somewhat ok
                    # to put our own files there if they don't exist already.
                    versioned_path = Path(root) / 'usr' / 'bin' / versioned_cmd
                    versioned_path.parent.mkdir(parents=True, exist_ok=True)
                    snippet = textwrap.dedent(f'''
                    #!/bin/sh

                    exec {unversioned_cmd} "$@"
                    ''').strip()
                    versioned_path.write_text(snippet)
                    versioned_path.chmod(0o755)

        def install_packages(root, packages):
            if packages:
                run_cmd(['apk', 'add', *sorted(set(packages))])
                fixup_llvm_tools(root, packages)

        def enable_network(root):
            shutil.copy('/etc/resolv.conf', reroot(root, '/etc/resolv.conf'))

        # Packages have already been installed, so we can speed things up a
        # bit
        if stage == 'init':
            version = list(map(str, version))
            minor = '.'.join(version[:2])
            version = '.'.join(version)
            url = _ALPINE_ROOTFS_URL.format(
                minor=minor,
                arch=alpine_arch,
                version=version,
            )

            with tempfile.NamedTemporaryFile(dir=path) as f:
                tar_path = Path(f.name).resolve()
                logger.info(f'Setting up Alpine chroot from {url} -> {tar_path}')

                with urllib.request.urlopen(url) as url, open(tar_path, 'wb') as f:
                    shutil.copyfileobj(url, f)

                with tarfile.open(tar_path, 'r') as f:
                    _tar_extractall(f, path=path)
            enable_network(path)
            install_packages(path, packages)
        elif stage == 'finalize':
            enable_network(path)
        else:
            raise ValueError(f'Unknown stage: {stage}')

    abi = abi or LISA_HOST_ABI
    use_qemu = abi != LISA_HOST_ABI

    qemu_msg = f' using QEMU userspace emulation to emulate {abi} on {LISA_HOST_ABI}' if use_qemu else ''
    logger.debug(f'Using Alpine v{".".join(map(str, version))} chroot with ABI {abi}{qemu_msg}.')

    # Check that QEMU userspace emulation is setup if we need it
    if use_qemu:
        qemu_arch = {
            'arm64': 'aarch64',
            'armeabi': 'arm',
            'armv7': 'arm',
        }.get(abi, abi)
        binfmt_path = Path('/proc/sys/fs/binfmt_misc/', f'qemu-{qemu_arch}')
        if not binfmt_path.exists():
            raise ValueError(f'Alpine chroot is setup for {qemu_arch} architecture but QEMU userspace emulation is not installed on the host (missing {binfmt_path})')

    dir_cache = DirCache(
        category='alpine_chroot',
        populate=populate,
    )
    alpine_arch =_abi_to_alpine_arch(abi)
    key = (
        version,
        alpine_arch,
        sorted(set(packages or [])),
    )
    chroot = dir_cache.get_entry(key)

    base_bind_paths = {
        '/proc': '/proc',
        '/sys': '/sys',
        '/dev': '/dev',
    }

    lowers = [chroot]
    if rust_spec:
        def populate_rust(key, path):
            rust_spec, *_ = key

            # Install Rust inside a folder that will be overlay-mounted at the
            # root of the chroot. The rustup_home and cargo_home asked by the
            # user will be the paths inside the chroot.
            homes = {
                reroot(path, v): v
                for k, v in rust_spec.items()
                if (
                    k in ('rustup_home', 'cargo_home', 'cargo_cache') and
                    v
                )
            }
            _bind_paths = {**base_bind_paths, **homes}
            try:
                # Mount the Rust install homes inside the chroot before
                # installing it, with all commands running inside the activated
                # chroot.
                mount_binds(chroot, _bind_paths)
                _install_rust(
                    rust_spec=rust_spec,
                    run_cmd=lambda cmd: run_cmd_chroot(chroot, cmd),
                )
            finally:
                mount_binds(chroot, _bind_paths, mount=False)

        rust_dir_cache = DirCache(
            category='rust_for_alpine_chroot',
            populate=populate_rust,
        )
        # Add the Alpine key to the Rust key, so that the Rust install is
        # allowed to depend on the state of the Alpine install (e.g. packages
        # being installed or not).
        rust_key = (rust_spec, key)
        rust_home = rust_dir_cache.get_entry(rust_key)
        lowers.append(rust_home)
        if (cargo_cache := rust_spec['cargo_cache']):
            base_bind_paths[reroot(rust_home, cargo_cache)] = cargo_cache

    # Bind a host path (key) to a path inside the chroot (value). Values have
    # to be absolute paths.
    bind_paths = {
        **base_bind_paths,
        **(bind_paths or {})
    }

    with _overlay_folders(lowers, backend=overlay_backend) as path:
        try:
            mount_binds(path, bind_paths)
            # We always need to "repopulate" the overlay in order to get a
            # working system with /etc/resolv.conf etc
            populate(key, path, stage='finalize')
            yield path
        except ContextManagerExit:
            mount_binds(path, bind_paths, mount=False)


def make_alpine_chroot(version, **kwargs):
    version = _resolve_alpine_version(version)
    return _make_alpine_chroot(version=version, **kwargs)


def _make_build_chroot_cmd(chroot, cmd):
    chroot = Path(chroot).resolve()
    cmd = ' '.join(map(quote, map(str, cmd)))
    # Source /etc/profile to get sane defaults for e.g. PATH. Otherwise, we
    # just inherit it from the host, which is broken.
    cmd = f'source /etc/profile && exec {cmd}'
    return ['chroot', chroot, 'sh', '-c', cmd]


@destroyablecontextmanager
def _overlay_folders(lowers, backend, upper=None, copy_filter=None):
    """
    Overlay folders on top of each other.

    :param lowers: List of read-only lower layers. The end of the list takes
        precedence. Apart from the first item in the list, all items must have
        been populated as an "upper" of its preceding lowers.
    :type lowers: list(str)

    :param upper: Read-write upper layer taking all the changes made to the
        mount point. If left out, a throw-away upper layer will be used.
    :type upper: str or None

    :param backend: Backend to use, one of:

        * ``overlayfs``: Uses Linux overlayfs mounts. This is the fastest and
          most space efficient method.
        * ``copy``: This uses plain copies to simulate overlayfs.
        * ``None``: defaults to ``overlayfs``.

        Note that mixing lowers created with different backends is not
        supported. Stick with the same backend when creating all the lowers in
        the stack.
    :type backend: str or None
    """
    logger = logging.getLogger(f'{__name__}.overlay')

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

        @destroyablecontextmanager
        def do_mount(dirs):
            dirs['lower'] = ':'.join(map(str, reversed(list(lowers))))
            cmd = [
                'mount',
                '-t', 'overlay', 'overlay',
                '-o', 'lowerdir={lower},workdir={work},upperdir={upper}'.format(**dirs),
                # Required on some setup, and recommended by the kernel doc:
                # https://docs.kernel.org/filesystems/overlayfs.html#user-xattr
                '-o', 'userxattr',
                # Having xino cannot be guaranteed, so we turn it off in order
                # to limit the variability of results based on the host setup.
                # This will make any related issue easier to reproduce on
                # another different setup.
                '-o', 'xino=off',
                '--',
                mount_point,
            ]
            _subprocess_log(cmd, logger=logger, level=logging.DEBUG)

            try:
                yield mount_point
            except ContextManagerExit:
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

        @destroyablecontextmanager
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

            # Apart from the first one in the list, all lowers are expected to
            # have been populated as an "upper" of the preceding lowers.
            # Therefore, we can simply make a copy of the top-most lower.
            #
            # This is the only way we can let the user delete a file in an
            # upper and subsequently restore that state.
            src = lowers[-1]
            _copytree(src=src, dst=mount_point)

            try:
                yield mount_point
            except ContextManagerExit:
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
            raise ValueError(f'Unknown overlay backend "{backend}"')

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
    def _get_key(self):
        """
        Return the checksum of the resource.
        """
        pass


class _FileOverlayBase(OverlayResource):
    """
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
    Base class for path-based overlays.
    """
    # This is racy with write_to(), but we are not trying to make something
    # really secure here, we just want to compute a unique token to be used as
    # a cache key
    def _get_key(self):
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
        content = content.encode('utf-8') if isinstance(content, str) else content
        self.content = content

    def write_to(self, dst):
        with open(dst, 'wb') as f:
            f.write(self.content)

    def _get_key(self):
        check = checksum(io.BytesIO(self.content), 'sha256')
        return f'{self.__class__.__name__}-{check}'

    def __str__(self):
        return f'{self.__class__.__qualname__}({self.content})'


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
        with tarfile.open(self.path) as f:
            _tar_extractall(f, dst)


class PatchOverlay(OverlayResource):
    """
    Patch to be applied on an existing file.

    :param overlay: Overlay providing the content of the patch.
    :type overlay: _FileOverlayBase
    """
    def __init__(self, overlay):
        self._overlay = overlay

    def write_to(self, dst):
        with tempfile.NamedTemporaryFile(mode='w+t') as patch:
            self._overlay.write_to(patch.name)
            subprocess_log(['patch', '-p0', '-r', '-', '-u', '--forward', dst, patch.name])

    def _get_key(self):
        """
        Return the checksum of the resource.
        """
        csum = self._overlay._get_key()
        return f'{self.__class__.__name__}-{csum}'

    def __str__(self):
        return f'{self.__class__.__qualname__}({self._overlay})'


class _KernelBuildEnvConf(SimpleMultiSrcConf):
    STRUCTURE = TopLevelKeyDesc('kernel-build-env-conf', 'Build environment settings',
        (
            KeyDesc('build-env', 'Environment used to build modules. Can be any of "alpine" (Alpine Linux chroot, recommended) or "host" (command ran directly on host system)', [typing.Literal['host', 'alpine']]),
            LevelKeyDesc('build-env-settings', 'build-env settings', (
                LevelKeyDesc('host', 'Settings for host build-env', (
                    KeyDesc('toolchain-path', 'Folder to prepend to PATH when executing toolchain command in the host build env. Toolchain autodetection will be restricted to that folder.', [str]),
                )),
                LevelKeyDesc('alpine', 'Settings for Alpine linux build-env', (
                    KeyDesc('version', 'Alpine linux version, e.g. 3.18.0', [None, str]),
                    KeyDesc('packages', 'List of Alpine linux packages to install. If that is provided, then errors while installing the package list provided by LISA will not raise an exception, so that the user can provide their own replacement for them. This allows future-proofing hardcoded package names in LISA, as Alpine package names might evolve between versions.', [None, typing.Sequence[str]]),
                )),
            )),

            KeyDesc('overlay-backend', 'Backend to use for overlaying folders while building modules. Can be "overlayfs" (overlayfs filesystem, recommended and fastest) or "copy (plain folder copy)', [str]),
            KeyDesc('make-variables', 'Extra variables to pass to "make" command, such as "CC"', [typing.Dict[str, object]]),

            VariadicLevelKeyDesc('modules', 'modules settings',
                LevelKeyDesc('<module-name>', 'For each module. The module shipped by LISA is "lisa"', (
                    KeyDesc('overlays', 'Overlays to apply to the sources of the given module', [typing.Dict[str, OverlayResource]]),
                )
            ))
        ),
    )

    DEFAULT_SRC = {
        'build-env': 'host',
        'overlay-backend': 'overlayfs',
    }

    def _get_key(self):
        return (
            self.get('overlay-backend'),
            self.get('build-env'),
            self.get('build-env-settings').to_map(),
            sorted(self.get('make-variables', {}).items()),
        )

    def _get_key_for_kmod(self, kmod):
        return (
            self._get_key(),
            sorted(
                (name, overlay._get_key())
                for name, overlay in self.get('modules', {}).get(kmod.mod_name, {}).get('overlays', {}).items()
            )
        )

    def _get_alpine_version(self):
        alpine_version = self['build-env-settings']['alpine'].get('version')
        return _resolve_alpine_version(alpine_version)


class _KernelBuildEnv(Loggable, SerializeViaConstructor):
    """
    :param path_cm: Context manager factory expected to return a path to a
        prepared kernel build env.
    :type path_cm: collections.abc.Callable

    :param build_conf: Build environment configuration. If specified as a
        string, it can be one of:

        * ``alpine``: Alpine linux chroot, providing a controlled
          environment
        * ``host``: No specific env is setup, whatever the host is using will
          be picked.
        * ``None``: defaults to ``host``.

        Otherwise, pass an instance of :class:`_KernelBuildEnvConf` of a mapping with
        the same structure.
    :type build_conf: collections.abc.Mapping or str or None
    """

    # Preserve checksum attribute when serializing, as it will allow hitting
    # the module cache without actually setting up the kernel build env in many
    # cases.
    _SERIALIZE_PRESERVED_ATTRS = {'checksum'}

    _KERNEL_ARCHIVE_URL_TEMPLATE = 'https://cdn.kernel.org/pub/linux/kernel/v{main_number}.x/linux-{version}.tar.xz'

    # We are only really interested in clang starting from version 13,
    # when the "musttail" return attribute was introduced.
    # On top of that, the kernel does not handle clang < 10.0.1
    _MIN_CLANG_VERSION = 11

    def __init__(self, path_cm, build_conf=None, target=None):
        self.conf, self.cc, self.cross_compile, self._cc_key, self.abi = self._resolve_conf(build_conf, target=target)

        self._make_path_cm = path_cm
        self._path_cm = None
        self.path = None
        self.checksum = None

    def _get_key_for_kmod(self, kmod):
        return (
            str(self._cc_key),
            *self.conf._get_key_for_kmod(kmod),
        )

    @classmethod
    def _resolve_conf(cls, conf, abi=None, target=None):
        def make_conf(conf):
            if isinstance(conf, _KernelBuildEnvConf):
                return conf
            elif conf is None:
                return _KernelBuildEnvConf()
            elif isinstance(conf, str):
                return _KernelBuildEnvConf.from_map({
                    'build-env': conf
                })
            elif isinstance(conf, Mapping):
                return _KernelBuildEnvConf.from_map(conf)
            else:
                raise TypeError(f'Unsupported value type for build_conf: {conf}')

        conf = make_conf(conf)
        make_vars, cc, cross_compile, cc_key, abi = cls._process_make_vars(conf, abi=abi, target=target)
        conf.add_src(src='processed make-variables', conf={'make-variables': make_vars})

        return (conf, cc, cross_compile, cc_key, abi)

    _SPEC_KEYS = ('path', 'checksum')

    def _to_spec(self):
        return {
            attr: getattr(self, attr)
            for attr in self._SPEC_KEYS
        }

    def _update_spec(self, spec):
        def update(x):
            val = spec.get(x)
            if val is not None:
                setattr(self, x, val)
        if spec:
            for attr in self._SPEC_KEYS:
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
                    if e.code == 404 and orig_version.rc is None:
                        try:
                            parts = decrement_version(parts)
                        except ValueError:
                            raise ValueError('Cannot fetch any tarball matching {orig_version}')

                    else:
                        raise e
                else:
                    cls._URL_CACHE[str(version)] = response.url
                    return (url, response)
        else:
            return (url, None)


    @classmethod
    def _prepare_tree(cls, path, cc, cross_compile, abi, build_conf, apply_overlays):
        logger = cls.get_logger()
        path = Path(path).resolve()
        toolchain_env = cls._make_toolchain_env_from_conf(build_conf)

        def make(*targets):
            return _kbuild_make_cmd(
                path=path,
                targets=targets,
                cc=cc,
                make_vars=build_conf.get('make-variables', {}),
            )

        def make_runner(cmd, allow_fail=False, env=None, **kwargs):
            _env = {
                **toolchain_env,
                **(env or {}),
            }
            def runner(amend_cmd=lambda cmd: cmd):
                _cmd = amend_cmd(cmd)

                try:
                    return _subprocess_log(
                        _cmd,
                        logger=logger,
                        level=logging.DEBUG,
                        env=_env,
                        **kwargs,
                    )
                except subprocess.CalledProcessError as e:
                    if allow_fail:
                        pretty_cmd = ' '.join(map(quote, map(str, cmd)))
                        logger.debug(f'Failed to run command {pretty_cmd}: {e}')
                        return ''
                    else:
                        raise e

            return runner

        def rewrite_runner(runner, amend_cmd):
            def new_runner(_amend_cmd=lambda cmd: cmd):
                __amend_cmd = lambda cmd: _amend_cmd(amend_cmd(cmd))
                return runner(__amend_cmd)
            return new_runner


        # Since GIT_CEILING_DIRECTORIES is colon-separated, we cannot
        # accept colons in the path
        assert ':' not in str(path)

        cmds = (
            # On non-host build env, we need to clean first, as binaries compiled
            # in e.g. scripts/ will probably not work inside the Alpine container,
            # since they would be linked against shared libraries on the host
            # system.
            #
            # On host build env, due to this bug:
            # https://lore.kernel.org/all/YfK18x%2FXrYL4Vw8o@syu-laptop/t/#md877c45455918f8c661dc324719b91a9906dc7a3
            # We need to get rid of vmlinux file in order to prevent the kernel
            # module build from generating split BTF information.
            # This issue should have been fixed by that patch, but requires
            # setting MODULE_ALLOW_BTF_MISMATCH=y which is not the default:
            # https://git.kernel.org/pub/scm/linux/kernel/git/bpf/bpf-next.git/commit/?id=5e214f2e43e4
            #
            # On top of that, a user-provided kernel tree with modules_prepare
            # already run would lead to not having e.g. some binaries stored in
            # the overlay. All would be well until the user does a manual
            # mrproper, at which point we are left hanging since we would hit
            # the cache, but would not be ready. "make modules" would probably
            # fix that but we could still have possibly issues with different
            # toolchains etc.
            (
                lambda path: make_runner(
                    ('git', '-c', 'core.checkStat=minimal', '-C', path, 'clean', '-fdx'),
                    allow_fail=True,
                    env={
                        **os.environ,
                        # Make sure we don't accidentally end up acting on a
                        # .git in e.g. $HOME if the kernel tree is not a git
                        # repo. This is especially important as we are using
                        # git clean.
                        'GIT_CEILING_DIRECTORIES': str(path),
                    },
                ),
                lambda path: make_runner(make('mrproper')),
            ),
            (
                lambda path: make_runner(make('olddefconfig', 'modules_prepare')),
            )
        )

        def fixup_kernel_build_env():
            # TODO: re-assess

            # The headers in /sys/kheaders.tar.xz generated by
            # CONFIG_IKHEADERS=y are broken since kernel/gen_kheaders.sh strip
            # some comments from the file. KBuild then proceeds on checking the
            # checksum and the check fails. This used to be a simple warning,
            # but has now been turned into an error in recent kernels.
            # We remove the SHA1 from the file so that the check is skipped.
            with contextlib.suppress(FileNotFoundError):
                def join(lines):
                    return b'\n'.join(lines)

                for _path in (path / 'include' / 'linux' / 'atomic').iterdir():
                    content = _path.read_bytes()
                    lines = content.split(b'\n')
                    i, last_line = [(i, line) for (i, line) in enumerate(lines) if line][-1]

                    if lines and last_line.lstrip().startswith(b'//'):
                        # Remove the last line, containing the sha1
                        without_sha1 = copy.copy(lines)
                        del without_sha1[i]
                        without_sha1 = join(without_sha1)
                        sha1 = hashlib.sha1(without_sha1).hexdigest()

                        # Update the sha1
                        updated = copy.copy(lines)
                        updated[i] = b'// ' + sha1.encode('ascii')
                        updated = join(updated)
                        _path.write_bytes(updated)

        build_env = build_conf['build-env']
        if build_env == 'alpine':
            settings = build_conf['build-env-settings']['alpine']
            version = settings.get('version', None)
            alpine_packages = settings.get('packages', None)
            make_vars = build_conf.get('make-variables', {})
            overlay_backend = build_conf['overlay-backend']

            @contextlib.contextmanager
            def cmd_cm(cmds):
                with _make_build_chroot(
                    cc=cc.name,
                    cross_compile=cross_compile,
                    abi=abi,
                    bind_paths={path: path},
                    overlay_backend=overlay_backend,
                    version=version,
                    packages=alpine_packages,
                ) as chroot:
                    yield [
                        [
                            rewrite_runner(
                                make_runner(path),
                                lambda cmd: _make_build_chroot_cmd(chroot, cmd)
                            )
                            for make_runner in runners
                        ]
                        for runners in cmds
                    ]
        elif build_env == 'host':
            @contextlib.contextmanager
            def cmd_cm(cmds):
                yield [
                    [
                        make_runner(path)
                        for make_runner in runners
                    ]
                    for runners in cmds
                ]
        else:
            raise ValueError('Unknown build-env kind: {build_env}')

        try:
            config_path = os.environ['KCONFIG_CONFIG']
        except KeyError:
            config_path = '.config'

        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = path / config_path

        try:
            config_content = config_path.read_bytes()
        except FileNotFoundError:
            config_content = None

        with cmd_cm(cmds) as _cmds:
            pre, post = _cmds
            logger.info(f'Preparing kernel tree for modules')

            for runner in pre:
                runner()

            # Ensure the configuration is available under .config, so that we
            # can rely on that. Overlays can now be applied to override it if
            # they need to. KCONFIG_CONFIG is set in _process_make_vars() to
            # ".config" so that all make commands run with these settings.
            if config_content:
                (path / '.config').write_bytes(config_content)

            # Apply the overlays before running make, so that it sees the
            # correct headers and conf etc
            apply_overlays()
            fixup_kernel_build_env()

            for runner in post:
                runner()

            # Re-apply the overlays, since we could have overwritten important
            # things, such as include/linux/vermagic.h
            apply_overlays()
            fixup_kernel_build_env()


    @classmethod
    def _process_make_vars(cls, build_conf, abi, target=None):
        logger = cls.get_logger()

        env = {
            k: str(v)
            for k, v in (
                (k, os.getenv(k)) for k in {
                    'CROSS_COMPILE',
                    'ARCH',
                    'KBUILD_MODPOST_WARN'
                }
            )
            if v is not None
        }

        make_vars = {
            **env,
            **dict(build_conf.get('make-variables', {}))
        }

        make_vars = {
            str(k): str(v)
            for k, v in make_vars.items()
        }

        # Force the value of .config, and we ensure somewhere else that we will
        # end up with the correct configuration there.
        make_vars['KCONFIG_CONFIG'] = '.config'

        try:
            arch = make_vars['ARCH']
        except KeyError:
            if abi:
                arch = _abi_to_kernel_arch(abi)
            else:
                raise ValueError('The ABI must be specified or the ARCH make variable')

        abi = abi or _kernel_arch_to_abi(arch)

        make_vars['ARCH'] = arch
        build_conf = build_conf.add_src(
            src='make-variables',
            conf={
                'make-variables': make_vars
            },
            inplace=False,
        )
        cc, cross_compile, cc_key = cls._resolve_toolchain(abi, build_conf, target=target)

        if 'clang' in cc.name and 'LLVM' not in make_vars:
            clang_version = _clang_version_static(cc.name)
            llvm_version = f'-{clang_version}' if clang_version else '1'
            make_vars['LLVM'] = llvm_version

        # Turn errors into warnings by default, as this otherwise prevents the
        # builds when the list of kernel symbols is not available.
        if 'KBUILD_MODPOST_WARN' not in make_vars:
            make_vars['KBUILD_MODPOST_WARN'] = '1'

        # Ensure the make vars contain the chosen compiler explicitly. It will
        # then be re-filtered right before invoking make to remove CC=gcc as it
        # can confuse KBuild.
        make_vars['CC'] = str(cc)
        if cross_compile:
            make_vars['CROSS_COMPILE'] = cross_compile


        # LLVM=0 is treated the same way as LLVM=1 by Kbuild, so we need to
        # remove it.
        if make_vars.get('LLVM') == '0':
            del make_vars['LLVM']

        llvm = make_vars.get('LLVM')

        # Some kernels have broken/old Kbuild that does not honor the LLVM=-N
        # suffixing, so force the suffixes ourselves.
        #
        # Also, the expectation of Kbuild in terms of binary name (e.g.
        # llvm-objcopy-17) are violated on Alpine that uses llvm17-objcopy
        # convention instead. So we override Kbuild detection with something
        # that works
        if build_conf['build-env'] == 'alpine':
            llvm_tool_name = alpine_llvm_tool_name
            if llvm:
                # TODO: Revisit:
                # Alpine does not ship multiple versions of e.g. lld, only
                # multiple versions of clang. Kbuild fails to find
                # ld.lld-<llvm_version> since that binary does not exist on
                # Alpine.
                #
                # Note from Alpine 3.21, there should be an ld.lld18 package in
                # addition to the main package, but then the main package
                # should be in version 19 anyway so there is probably no point
                # in supporting that. From some comments in the build recipe,
                # that ld.lld18 package is only there for zig, so there is a
                # good chance it disappears again in 3.22
                make_vars.setdefault('LD', 'ld.lld')
                make_vars.setdefault('HOSTLD', 'ld.lld')
        else:
            llvm_tool_name = default_llvm_tool_name

        if llvm and llvm.startswith('-'):
            updated = {
                var: llvm_tool_name(_var.lower(), llvm)
                for _var in (
                    tool.upper()
                    for tool in _KERNEL_BINUTILS
                )
                for var in (_var, f'HOST{_var}')
            }
            make_vars = {**updated, **make_vars}

        assert 'ARCH' in make_vars

        def log_fragment(var):
            val = make_vars.get(var)
            fragment = f'{var}={val}' if val is not None else ''
            return fragment

        variables = ', '.join(filter(bool, map(log_fragment, ('CC', 'CROSS_COMPILE', 'LLVM', 'ARCH'))))
        logger.info(f'Toolchain detected: {variables}')
        return (make_vars, cc, cross_compile, cc_key, abi)

    @classmethod
    def _make_toolchain_env(cls, toolchain_path=None, env=None):
        env = env or os.environ
        if toolchain_path is not None:
            assert toolchain_path
            path = env.get('PATH', '')
            env = {
                **env,
                'PATH': ':'.join((toolchain_path, path))
            }

        return {**os.environ, **env}

    @classmethod
    def _make_toolchain_env_from_conf(cls, build_conf):
        if build_conf['build-env'] == 'host':
            toolchain_path = build_conf['build-env-settings']['host'].get('toolchain-path')
            env = {'PATH': HOST_PATH}
        else:
            env = {}
            toolchain_path = None
        return cls._make_toolchain_env(toolchain_path, env=env)

    @classmethod
    def _check_cc_version(cls, cc, toolchain_path):
        if 'clang' in cc.name:
            env = cls._make_toolchain_env(toolchain_path)
            try:
                major, *_ = _clang_version(cc, env=env)
            except ValueError:
                pass
            else:
                if major >= cls._MIN_CLANG_VERSION:
                    return True
        else:
            return True

        return False

    @classmethod
    def _resolve_toolchain(cls, abi, build_conf, target=None):
        logger = cls.get_logger()
        env = cls._make_toolchain_env_from_conf(build_conf)

        def get_cross_compile_prio(cross_compile):
            # Android kernels are typically built with and android target
            # triplet. Failing to use the correct toolchain may result in
            # compilation errors such as non-recognized CLI options, even if
            # the toolchain binaries are the same.
            if target is not None and isinstance(target.target, AndroidTarget):
                return 0 if 'android' in cross_compile else 1
            else:
                return 0

        def priority_to(cc):
            def prio(build_env, _cc, _cross_compile):
                cc_prio = 0 if (cc and cc in _cc.name) else 1
                return (
                    cc_prio,
                    get_cross_compile_prio(_cross_compile)
                )
            return prio

        # If we can't get more precise info, select anything we can
        cc_priority = priority_to(None)

        if target:
            config = target.config.typed_config
            if config.get('CONFIG_CC_IS_GCC', False):
                cc_priority = priority_to('gcc')
            elif config.get('CONFIG_CC_IS_CLANG', False):
                clang_version = config.get('CONFIG_CLANG_VERSION', None)
                if clang_version is None:
                    cc_priority = priority_to('clang')
                else:
                    clang_version = clang_version // 10_000

                    def cc_priority(build_env, cc, cross_compile):
                        def prio(cc):
                            # Firstly, we give priority to anything that is
                            # clang.
                            class Kind(IntEnum):
                                IS_CLANG = 0
                                NOT_CLANG = 1

                            # We then choose higher versions over lower
                            # versions as there is more chance to have all the
                            # required features and C extensions.
                            class Version(IntEnum):
                                HIGHER_VERSION = 0
                                LOWER_VERSION = 1
                                UNKNOWN_VERSION = 2

                            # As a tie-breaker, if we have to choose between
                            # "clang" and "clang-XYZ" when both --version
                            # report to be XYZ, use "clang".

                            if build_env == 'host':
                                # This improves compat with GKI prebuilt
                                # toolchains that ships a mix of clang-XYZ and
                                # clang binaries, but only unversioned names
                                # for the rest of the tools.
                                class Name(IntEnum):
                                    UNVERSIONED_NAME = 0
                                    VERSIONED_NAME = 1
                            elif build_env == 'alpine':
                                # On Alpine, we always want to pick the named
                                # version if we can, as it will always match
                                # better what we want if it is available.
                                class Name(IntEnum):
                                    VERSIONED_NAME = 0
                                    UNVERSIONED_NAME = 1
                            else:
                                raise ValueError(f'Unknown build environment: {build_env}')

                            def version_key(version):
                                if version:
                                    return (
                                        Version.HIGHER_VERSION if version >= clang_version else Version.LOWER_VERSION,
                                        # Try the versions closest to the one we
                                        # want
                                        abs(clang_version - version)
                                    )
                                else:
                                    return (Version.UNKNOWN_VERSION,)

                            if 'clang' in cc.name:
                                version = re.search(r'[0-9]+', cc.name)
                                if version is None:
                                    convention = Name.UNVERSIONED_NAME
                                    if build_env == 'host':
                                        try:
                                            version, *_ = _clang_version(cc, env=env)
                                        except ValueError:
                                            version = None
                                    else:
                                        # This is the only choice since the
                                        # file name gives no clue and we cannot
                                        # run the binary (by definition, as it
                                        # is not the host environment).
                                        version = None
                                else:
                                    convention = Name.VERSIONED_NAME
                                    version = int(version.group(0))

                                return (Kind.IS_CLANG, version_key(version), convention)
                            else:
                                return (Kind.NOT_CLANG,)

                        return (
                            prio(cc),
                            get_cross_compile_prio(cross_compile)
                        )
            else:
                try:
                    proc_version = target.read_value('/proc/version')
                except TargetStableError:
                    pass
                else:
                    proc_version = proc_version.lower()
                    if 'gcc' in proc_version:
                        cc_priority = priority_to('gcc')
                    if 'clang' in proc_version:
                        cc_priority = priority_to('clang')

        make_vars = build_conf.get('make-variables', {})

        try:
            cross_compiles = [make_vars['CROSS_COMPILE']]
        except KeyError:
            try:
                cross_compiles = [os.environ['CROSS_COMPILE']]
            except KeyError:
                if abi == 'arm64':
                    cross_compiles = ['aarch64-linux-gnu-', 'aarch64-none-linux-gnu-', 'aarch64-none-elf-', 'aarch64-linux-android-', 'aarch64-none-linux-android-']
                elif abi == 'armeabi':
                    cross_compiles = ['arm-linux-gnueabi-', 'arm-none-linux-gnueabi-', 'arm-linux-eabi-', 'arm-none-linux-eabi-', 'arm-none-eabi-']
                elif abi == 'x86':
                    cross_compiles = ['i686-linux-gnu-']
                else:
                    cross_compiles = ['']
                    if abi != LISA_HOST_ABI:
                        logger.error(f'ABI {abi} not recognized, CROSS_COMPILE env var needs to be set')

                logger.debug(f'CROSS_COMPILE env var not set, assuming "{cross_compiles}"')

        if abi == LISA_HOST_ABI:
            cross_compiles.insert(0, '')

        cross_compiles = cross_compiles or ['']
        cross_compiles = deduplicate(cross_compiles, keep_last=False)

        # The format of "ccs" dict is:
        # (CC=, CROSS_COMPILE=): <binary name>
        ccs = [
            *(
                (Path(f'clang-{i}'), cross_compile)
                # Cover for the next 10 years starting from 2021
                for i in reversed(range(
                    cls._MIN_CLANG_VERSION,
                    cls._MIN_CLANG_VERSION + 10 * 2
                ))
                for cross_compile in cross_compiles
            ),
            *(
                (Path('clang'), cross_compile)
                for cross_compile in cross_compiles
            ),
            *(
                (Path('gcc'), cross_compile)
                for cross_compile in cross_compiles
            ),
        ]

        if 'CC' in make_vars:
            _cc = _make_vars_cc(make_vars)
            ccs = [
                (_cc, cross_compile)
                for cross_compile in cross_compiles
            ]

        if 'LLVM' in make_vars:
            llvm = make_vars['LLVM']
            _cc = _make_vars_cc(make_vars, 'clang')
            llvm_version = llvm if llvm.startswith('-') else None
            if _cc.name == 'clang' and llvm_version:
                _cc = _cc.with_name(f'{_cc.name}{llvm_version}')
                ccs = [
                    (_cc, cross_compile)
                    for cross_compile in cross_compiles
                ]

        # Give priority for the toolchain the kernel seem to have been compiled
        # with
        def key(item):
            (cc, cross_compile) = item
            return cc_priority(build_conf['build-env'], cc, cross_compile)

        ccs = deduplicate(ccs, keep_last=False)
        ccs = sorted(ccs, key=key)

        cc = None
        cross_compile = None
        cc_key = None

        # Only run the check on host build env, as other build envs are
        # expected to be correctly configured.
        if build_conf['build-env'] == 'host':

            def cc_cmd(cc, cross_compile, opts):
                if 'gcc' in cc.name:
                    return (cc.with_name(f'{cross_compile}{cc}'), *opts)
                elif 'clang' in cc.name:
                    return (cc, *([f'--target={cross_compile}'] if cross_compile else []), *opts)
                else:
                    raise ValueError(f'Cannot test presence of compiler "{cc}"')

            def test_cmd(cc, cross_compile):
                opts = ('-x' 'c', '-c', '-', '-o', '/dev/null')
                return cc_cmd(cc, cross_compile, opts)

            def version_cmd(cc, cross_compile):
                opts = ('--version',)
                return cc_cmd(cc, cross_compile, opts)

            toolchain_path = build_conf['build-env-settings']['host'].get('toolchain-path', None)

            def is_in_toolchain_path(cc, cross_compile):
                if toolchain_path is None:
                    return True
                else:
                    cmd = cc_cmd(cc, cross_compile, opts=[])
                    bin_, *_ = cmd
                    bin_ = shutil.which(bin_, path=toolchain_path)
                    return bin_ is not None

            ccs = [
                (cc, cross_compile)
                for cc, cross_compile in ccs
                if is_in_toolchain_path(cc, cross_compile)
            ]

            for (cc, cross_compile) in ccs:
                cmd = test_cmd(cc, cross_compile)

                pretty_cmd = ' '.join(map(quote, map(str, cmd)))
                try:
                    subprocess.check_output(
                        cmd,
                        # Most basic compiler input that will not do anything.
                        input=b';',
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                except subprocess.CalledProcessError as e:
                    logger.debug(f'Checking {cc} compiler: {pretty_cmd} failed with:\n{e.output.decode()}')
                    continue
                except FileNotFoundError as e:
                    logger.debug(f'Checking {cc} compiler: {e}')
                    continue
                else:
                    if cls._check_cc_version(cc, toolchain_path):
                        cc_key = subprocess.check_output(
                            version_cmd(cc, cross_compile),
                            env=env,
                        )
                        break
            else:
                cross = ' or '.join(
                    f'CROSS_COMPILE={cross_compile}'
                    for cross_compile in cross_compiles
                )
                try:
                    cc = _make_vars_cc(make_vars)
                except KeyError:
                    cc = None
                with_cc = f' with CC={cc}' if cc else ''
                raise ValueError(f'Could not find a working toolchain for {cross}{with_cc}')

        elif build_conf['build-env'] == 'alpine':
            alpine_version = build_conf._get_alpine_version()

            if ccs:
                for (cc, cross_compile) in ccs:
                    try:
                        _find_alpine_cc_packages(
                            # We check against the package list for the host
                            # ABI, assuming we will not need emulation to run
                            # the toolchain.
                            abi=LISA_HOST_ABI,
                            version=tuple(alpine_version),
                            cc=cc.name,
                            cross_compile=cross_compile,
                        )
                    except ValueError:
                        pass
                    else:
                        break
                else:
                    ccs, *_ = zip(*ccs)
                    ccs = ', '.join(map(str, sorted(ccs)))
                    alpine_version = '.'.join(map(str, alpine_version))
                    raise ValueError(f'None of the considered toolchains are available on Alpine Linux v{alpine_version}: {ccs}')

        if cc is None:
            raise ValueError(f'Could not detect which compiler to use for CC')

        if cross_compile is None:
            raise ValueError(f'Could not detect which CROSS_COMPILE value to use')

        ideal_cc, ideal_cross_compile = ccs[0]
        if str(cc) != str(ideal_cc) or ideal_cross_compile != cross_compile:
            logger.warning(f'Could not find ideal CC={ideal_cc} and CROSS_COMPILE={ideal_cross_compile} but found CC={cc} and CROSS_COMPILE={cross_compile} instead. Results may vary from working fine to crashing the kernel')

        return (cc, cross_compile, cc_key)

    @classmethod
    @SerializeViaConstructor.constructor
    def from_target(cls, target, tree_path=None, cache=True, build_conf=None):
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

        :param cache: If ``True``, will attempt to cache intermediate steps.
        :type cache: bool

        :param build_conf: See :class:`lisa._kmod._KernelBuildEnv`.
        :type build_conf: str or None
        """
        plat_info = target.plat_info
        abi = plat_info['abi']
        kernel_info = plat_info['kernel']

        build_conf, cc, cross_compile, cc_key, _abi = cls._resolve_conf(build_conf, abi=abi, target=target)
        assert _abi == abi

        class LoaderNotSelected(ValueError):
            pass

        @contextlib.contextmanager
        def from_installed_headers():
            """
            Get the kernel tree from /lib/modules
            """
            if build_conf['build-env'] == 'alpine':
                raise LoaderNotSelected(f'Building from /lib/modules is not supported with the Alpine build environment as /lib/modules might not be self contained (i.e. symlinks pointing outside)')
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
                    if cc.name in proc_version:
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
                            raise LoaderNotSelected(f'{target_path} is not a folder')
                    else:
                        raise LoaderNotSelected(f'The chosen compiler ({cc}) is different from the one used to build the kernel ({proc_version}), /lib/modules/ tree will not be used')
                else:
                    raise LoaderNotSelected(f'Building from /lib/modules/.../build/ is only supported for local targets')

        @contextlib.contextmanager
        def _from_target_sources(pull, **kwargs):
            """
            Overlay some content taken from the target on the user tree, such
            as /proc/config.gz
            """
            version = kernel_info['version']
            with tempfile.TemporaryDirectory() as temp:
                temp = Path(temp)
                overlays = pull(target, temp)

                with cls.from_overlays(
                    version=version,
                    overlays=overlays,
                    cache=cache,
                    tree_path=tree_path,
                    build_conf=build_conf,
                    **kwargs,
                ) as tree:
                    yield tree._to_spec()

        def missing_configs(configs):
            configs = ' and '.join(
                f'{conf}=y'
                for conf in configs
            )
            return LoaderNotSelected(f'Needs {configs}')

        def from_sysfs_headers():
            """
            From /sys/kernel/kheaders.tar.xz and /proc/config.gz
            """
            def pull(target, temp):
                try:
                    target.cached_pull('/proc/config.gz', str(temp), as_root=True)
                except Exception:
                    raise missing_configs(('CONFIG_IKCONFIG_PROC',))


                target_kernel_sha1 = target.plat_info['kernel']['version'].sha1
                if target_kernel_sha1 is None:
                    kheaders_key = None
                else:
                    with open(temp / 'config.gz', 'rb') as f:
                        config_checksum = checksum(f, 'sha256')
                    kheaders_key = (
                        target_kernel_sha1,
                        config_checksum,
                    )

                try:
                    target.cached_pull(
                        '/sys/kernel/kheaders.tar.xz',
                        str(temp),
                        via_temp=True,
                        as_root=True,
                        key=kheaders_key,
                    )
                except Exception:
                    raise missing_configs(('CONFIG_IKHEADERS',))

                return {
                    # We can use .config as we control KCONFIG_CONFIG in _process_make_vars()
                    FileOverlay.from_path(temp / 'config.gz', decompress=True): '.config',
                    TarOverlay.from_path(temp / 'kheaders.tar.xz'): '.',
                }

            return _from_target_sources(pull)

        def from_proc_config():
            """
            From /proc/config.gz
            """
            def pull(target, temp):
                try:
                    target.cached_pull('/proc/config.gz', str(temp), as_root=True)
                except Exception:
                    raise missing_configs(('CONFIG_IKCONFIG_PROC',))

                return {
                    # We can use .config as we control KCONFIG_CONFIG in _process_make_vars()
                    FileOverlay.from_path(temp / 'config.gz', decompress=True): '.config',
                }

            return _from_target_sources(pull)

        @contextlib.contextmanager
        def from_user_tree():
            """
            Purely from the tree passed by the user.
            """
            if tree_path is None:
                raise LoaderNotSelected('Use tree_path != None to build from a user-provided tree')
            else:
                # We still need to run make modules_prepare on the provided
                # tree
                with cls.from_overlays(
                    tree_path=tree_path,
                    version=kernel_info['version'],
                    cache=cache,
                    build_conf=build_conf,
                ) as tree:
                    yield tree._to_spec()

        @contextlib.contextmanager
        def try_loaders(loaders):
            logger = cls.get_logger()
            exceps = []
            for loader in loaders:
                logger.debug(f'Trying to load kernel tree using loader {loader.__name__} ...')
                try:
                    cm = loader()
                    spec = cm.__enter__()
                except Exception as e:
                    logger.debug(f'Failed to load kernel tree using loader {loader.__name__}: {e.__class__.__name__}: {e}')
                    # If the exception is coming from the guts of the
                    # machinery, we want a backtrace for easier debugging.
                    if not isinstance(e, LoaderNotSelected):
                        logger.debug(
                            ''.join(traceback.format_tb(e.__traceback__))
                        )
                    exceps.append((loader, e))
                else:
                    logger.debug(f'Loaded kernel tree using loader {loader.__name__}')
                    try:
                        yield spec
                    except Exception as e:
                        cm.__exit__(type(e), e, e.__traceback__)
                        raise
                    else:
                        cm.__exit__(None, None, None)
                        return

            excep_str = "\n".join(
                f"{loader.__name__}: {e.__class__.__name__}: {e}"
                for loader, e in exceps
            )
            raise ValueError(f'Could not load kernel trees:\n{excep_str}')

        # Try these loaders in the given order, until one succeeds
        loaders = [from_installed_headers, from_sysfs_headers, from_proc_config, from_user_tree]

        return cls(
            path_cm=functools.partial(try_loaders, loaders),
            build_conf=build_conf,
            target=target,
        )

    @classmethod
    @SerializeViaConstructor.constructor
    def from_path(cls, path, cache=True, build_conf=None):
        """
        Build a tree from the given ``path`` to sources.
        """
        return cls.from_overlays(
            tree_path=path,
            cache=cache,
            build_conf=build_conf,
        )

    @classmethod
    @SerializeViaConstructor.constructor
    def from_overlays(cls, version=None, tree_path=None, overlays=None, cache=True, build_conf=None):
        """
        Build a tree from the given overlays, to be applied on a source tree.

        :param version: Version of the kernel to be used.
        :type version: devlib.target.KernelVersion or str

        :param overlays: List of overlays to apply on the tree.
        :type overlays: list(OverlayResource)
        """
        logger = cls.get_logger()
        overlays = overlays or {}
        build_conf, cc, cross_compile, cc_key, abi = cls._resolve_conf(build_conf)

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
            for overlay, dst in overlays.items():
                logger.debug(f'Unpacking overlay {overlay} -> {dst}')
                overlay.write_to(os.path.join(path, dst))

        def prepare_overlay(path):
            cls._prepare_tree(
                path,
                cc=cc,
                cross_compile=cross_compile,
                abi=abi,
                build_conf=build_conf,
                apply_overlays=functools.partial(apply_overlays, path),
            )

        @contextlib.contextmanager
        def overlay_cm(args):
            base_path, tree_key = args
            base_path = Path(base_path).resolve()
            overlay_backend = build_conf['overlay-backend']

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
                        overlay._get_key()
                        for overlay, dst in overlays.items()
                    ) + [
                        tree_key,
                        str(cc_key),
                        build_conf._get_key(),
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
                    dir_cache.get_entry(url),
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
            build_conf=build_conf,
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

            with tarfile.open(tar_path) as f:
                # Account for a top-level folder in the archive
                prefix = os.path.commonpath(
                    member.path
                    for member in f.getmembers()
                )
                _tar_extractall(f, extract_folder)
        except Exception:
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

    _RUST_SPEC = None
    """
    Rust version and components to install in the build environment.
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

        self.logger.debug(f'Created {self.__class__.__qualname__} with name {self._mod_name} and sources: {", ".join(self.src.keys())}')

    @property
    def code_files(self):
        return {
            name: content
            for name, content in self.src.items()
            if any(
                name.endswith(extension)
                for extension in ('.c', '.h', '.rs')
            )
        }

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
        Checksum of the module's sources & Makefile.
        """
        def checksum(content):
            m = hashlib.sha1()
            content = content if isinstance(content, bytes) else content.encode('utf-8')
            m.update(content)
            return m.hexdigest()

        content = sorted(
            (checksum(content), name)
            for name, content in self.src.items()
        )

        # Recreate the output of sha1sum over multiple files, and checksum
        # that.
        content = '\n'.join(
            f'{csum}  ./{name}'
            for csum, name in content
        ) + '\n'

        return checksum(content)

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

    def compile(self, kernel_build_env, make_vars=None):
        """
        Compile the module and returns the ``bytestring`` content of the
        ``.ko`` file.

        :param kernel_build_env: kernel build env to build the module against.
        :type kernel_build_env: _KernelBuildEnv

        :param make_vars: Variables passed on ``make`` command line. This can
            be used for variables only impacting the module, otherwise it's
            better to set them when creating the ``kernel_build_env``.
        :type make_vars: dict(str, object) or None
        """
        make_vars = {
            **kernel_build_env.conf.get('make-variables', {}),
            **(make_vars or {}),
        }
        cc = kernel_build_env.cc
        cross_compile = kernel_build_env.cross_compile
        abi = kernel_build_env.abi
        tree_path = Path(kernel_build_env.path)
        # "inherit" the build env from the _KernelBuildEnv as we must use the same
        # environment as what was used for "make modules_prepare"
        build_conf = kernel_build_env.conf
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
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(content)

        def make_cmd(tree_path, mod_path, make_vars):
            make_vars = {
                **make_vars,
                **dict(
                    M=mod_path,
                    LISA_KMOD_NAME=self.mod_name,
                    KERNEL_SRC=tree_path,
                    MODULE_SRC=mod_path,
                    MODULE_OBJ=mod_path,
                )
            }
            return _kbuild_make_cmd(
                cc=cc,
                path=tree_path,
                targets=['modules'],
                make_vars=make_vars,
            )
            return cmd

        def find_mod_file(path):
            filenames = glob.glob(str(path.resolve() / '*.ko'))

            if not filenames:
                raise FileNotFoundError(f'Could not find .ko file in {path}')
            elif len(filenames) > 1:
                raise ValueError(f'Found more than one .ko file in {path}: {filenames}')
            else:
                return filenames[0]

        @contextlib.contextmanager
        def cargo_target_dir(cargo_cache):
            if cargo_cache:
                target_dir = cargo_cache / 'reference'
                tmp_target_dir = cargo_cache / f'tmp_{uuid.uuid4().hex}'

                # Move the build cache to a temporary location, so we know no
                # one else will be touching it while we are using it. Cargo
                # protects itself from concurrent accesses to the target
                # folder, but this does not extend to all the steps we do after
                # cargo has run to use the build artifacts.
                #
                # Path.rename() gives the same guarantees as os.rename(), and
                # on POSIX platforms this is an atomic operation if they are on
                # the same filesystem.
                try:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target_dir.rename(tmp_target_dir)
                except Exception:
                    tmp_target_dir = None
            else:
                tmp_target_dir = None

            try:
                yield tmp_target_dir
            finally:
                if tmp_target_dir:
                    #  When we are finished with the CARGO_TARGET_DIR, we
                    #  place it back at the expected location for the next
                    #  build to find it.
                    try:
                        tmp_target_dir.rename(target_dir)
                    except Exception:
                        # If we could not promote it back to the reference
                        # CARGO_TARGET_DIR, there is no point in keeping it
                        # around.
                        shutil.rmtree(tmp_target_dir)

        if build_conf['build-env'] == 'alpine':
            settings = build_conf['build-env-settings']['alpine']
            alpine_version = settings.get('version', None)
            alpine_packages = settings.get('packages', None)

            @contextlib.contextmanager
            def rust_cm(rust_spec):
                if rust_spec:
                    rust_home = Path('/opt/rust')
                    rustup_home = rust_home / 'rustup'
                    cargo_home = rust_home / 'cargo'
                    cargo_cache = rust_home / 'cargo_cache'

                    _rust_spec = {
                        **rust_spec,
                        'rustup_home': rustup_home,
                        'cargo_home': cargo_home,
                        'cargo_cache': cargo_cache,
                    }
                    rust_env = {
                        'RUSTUP_HOME': rustup_home,
                        'CARGO_HOME': cargo_home,
                        'RUST_VERSION': rust_spec['version'],
                    }

                    @contextlib.contextmanager
                    def _cargo_target_dir(chroot, rust_spec, rust_env):
                        try:
                            cargo_cache = rust_spec['cargo_cache']
                        except KeyError:
                            cargo_cache = None
                        else:
                            cargo_cache = reroot(chroot, cargo_cache)

                        with cargo_target_dir(cargo_cache) as target_dir:
                            yield {
                                **rust_env,
                                'CARGO_TARGET_DIR': target_dir,
                            }

                    yield (_rust_spec, rust_env, _cargo_target_dir)
                else:
                    yield (None, {}, lambda chroot, rust_spec, rust_env: nullcontext(rust_env))

            @contextlib.contextmanager
            def cmd_cm(rust_spec):
                with rust_cm(rust_spec) as (_rust_spec, rust_env, _cargo_target_dir):
                    with _make_build_chroot(
                        cc=cc.name,
                        cross_compile=cross_compile,
                        bind_paths=bind_paths,
                        abi=abi,
                        overlay_backend=build_conf['overlay-backend'],
                        version=alpine_version,
                        packages=alpine_packages,
                        rust_spec=_rust_spec,
                    ) as chroot:
                        # Do not use a CM here to avoid choking on permission
                        # issues. Since the chroot itself will be entirely
                        # removed it's not a problem.
                        mod_path = Path(tempfile.mkdtemp(dir=chroot / 'tmp'))

                        with _cargo_target_dir(chroot, _rust_spec, rust_env) as _rust_env:
                            cmd = make_cmd(
                                tree_path=tree_path,
                                mod_path=f'/{mod_path.relative_to(chroot)}',
                                make_vars={
                                    **make_vars,
                                    **_rust_env,
                                }
                            )
                            yield (mod_path, _make_build_chroot_cmd(chroot, cmd))

        elif build_conf['build-env'] == 'host':
            def install_rust(rust_spec):
                def populate(key, path):
                    rust_spec = {
                        **dict(key),
                        'rustup_home': path / 'rustup',
                        'cargo_home': path / 'cargo',
                        'cargo_cache': path / 'cargo_cache',
                    }
                    _install_rust(
                        rust_spec=rust_spec,
                        run_cmd=lambda cmd: _subprocess_log(cmd, logger=logger, level=logging.DEBUG)
                    )

                dir_cache = DirCache(
                    category='rust_home',
                    populate=populate,
                )
                key = sorted(rust_spec.items())
                rust_home = dir_cache.get_entry(key)
                return rust_home

            @contextlib.contextmanager
            def rust_cm(rust_spec):
                if rust_spec:
                    rust_home = install_rust(rust_spec)
                    cargo_cache = rust_home / 'cargo_cache'
                    with cargo_target_dir(cargo_cache) as target_dir:
                        yield {
                            'RUSTUP_HOME': rust_home / 'rustup',
                            'CARGO_HOME': rust_home / 'cargo',
                            'CARGO_TARGET_DIR': target_dir,
                            'RUST_VERSION': rust_spec['version'],
                        }
                else:
                    yield {}

            @contextlib.contextmanager
            def cmd_cm(rust_spec):
                with tempfile.TemporaryDirectory() as mod_path:
                    with rust_cm(rust_spec) as rust_env:
                        cmd = make_cmd(
                            tree_path=tree_path,
                            mod_path=mod_path,
                            make_vars={
                                **make_vars,
                                **rust_env,
                            },
                        )

                        yield (mod_path, cmd)
        else:
            raise ValueError('Unknown build-env kind: {build_env}')

        rust_spec = self._RUST_SPEC or {}
        env = _KernelBuildEnv._make_toolchain_env_from_conf(build_conf)
        with cmd_cm(rust_spec) as (mod_path, cmd):
            mod_path = Path(mod_path)
            populate_mod(mod_path)

            logger.info(f'Compiling kernel module {self.mod_name}')
            _subprocess_log(cmd, logger=logger, level=logging.DEBUG, env=env)

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
            root = Path(root)
            if root.name == '__pycache__':
                return
            else:
                for f in files:
                    yield (root / f)

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

    :param kernel_build_env: Kernel source tree to use to build the module against.
    :type kernel_build_env: lisa._kmod._KernelBuildEnv
    """
    def __init__(self, target, src, kernel_build_env=None):
        self.target = target

        if not isinstance(kernel_build_env, _KernelBuildEnv):
            kernel_build_env = _KernelBuildEnv.from_target(
                target=self.target,
                tree_path=kernel_build_env,
            )

        self._kernel_build_env = kernel_build_env

        mod_name = src.mod_name
        logger = self.logger
        overlays = kernel_build_env.conf.get('modules', {}).get(mod_name, {}).get('overlays', {})

        def apply_overlay(src, name, overlay):
            try:
                content = src.src[name]
            except KeyError:
                pass
            else:
                logger.debug(f'Applying patch to module {mod_name}, file {name}: {overlay}')
                with tempfile.NamedTemporaryFile(suffix=name) as f:
                    path = Path(f.name)
                    path.write_bytes(content)
                    overlay.write_to(path)
                    src.src[name] = path.read_bytes()

        src = copy.deepcopy(src)
        if overlays:
            for name, overlay in overlays.items():
                apply_overlay(src, name, overlay)

        self.src = src

    @property
    def mod_name(self):
        return self.src.mod_name

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
    def kernel_build_env(self):
        tree = self._kernel_build_env
        arch = _abi_to_kernel_arch(
            self.target.plat_info['abi']
        )
        tree_arch = tree.conf['make-variables']['ARCH']
        if tree_arch != arch:
            raise ValueError(f'The kernel build env ({tree_arch}) was not prepared for the same architecture as the target ({arch}). Please set ARCH={arch} make variable.')
        else:
            return tree

    @property
    def _compile_needs_root(self):
        tree = self.kernel_build_env
        return (
            tree.conf['build-env'] != 'host' or
            tree.conf['overlay-backend'] == 'overlayfs'
        )

    # Dummy memoized wrapper. The only reason we need one is that _do_compile()
    # needs to be pickleable to be sent to a multiprocessing Process, so it
    # cannot be overriden by a wrapper

    def _compile(self, make_vars=None):
        make_vars = make_vars or {}
        return self._memoized_compile(make_vars=tuple(sorted(make_vars.items())))

    @memoized
    def _memoized_compile(self, make_vars):
        make_vars = dict(make_vars)

        compile_ = self._do_compile.__func__
        if self._compile_needs_root:
            compile_ = ensure_root(compile_, inline=True)

        bin_, spec = compile_(self, make_vars=make_vars)
        # Get back _KernelBuildEnv._to_spec() and update the _KernelBuildEnv we have in
        # this process with it to remember the checksum, in case ensure_root()
        # spawned a new process. This is then used by Target.get_kmod() that
        # will reinject the known spec when creating new modules from the
        # default _KernelBuildEnv
        self.kernel_build_env._update_spec(spec)
        return bin_

    def _do_compile(self, make_vars=None):

        kernel_build_env = self.kernel_build_env
        extra_make_vars = make_vars or {}
        all_make_vars = {
            **kernel_build_env.conf.get('make-variables', {}),
            **extra_make_vars,
        }
        src = self.src

        def get_key(kernel_build_env):
            kernel_checksum = kernel_build_env.checksum
            if kernel_checksum is None:
                raise ValueError('kernel build env has no checksum')
            else:
                key = (
                    kernel_checksum,
                    kernel_build_env._get_key_for_kmod(self),
                    src.checksum,
                    all_make_vars,
                )
                return key

        def get_bin(kernel_build_env):
            return src.compile(
                kernel_build_env=kernel_build_env,
                make_vars=extra_make_vars,
            )

        def lookup_cache(kernel_build_env, key, enter_cm=False):
            cm = kernel_build_env if enter_cm else nullcontext(kernel_build_env)

            def populate(key, path):
                with cm as kernel_build_env:
                    with open(path / 'mod.ko', 'wb') as f:
                        f.write(get_bin(kernel_build_env))

            dir_cache = DirCache(
                category='kernel_modules',
                populate=populate,
            )
            cache_path = dir_cache.get_entry(key)
            with open(cache_path / 'mod.ko', 'rb') as f:
                return f.read()

        # First try on the "bare" kernel build env, i.e. before calling __enter__().
        # If this happens to have enough information to hit the cache, we just
        # avoided a possibly costly setup of compilation environment
        try:
            key = get_key(kernel_build_env)
        except ValueError:
            with kernel_build_env as kernel_build_env:
                if kernel_build_env.checksum is None:
                    # Only cache the module if the kernel build env has a defined
                    # checksum, which is not always the case when it's not
                    # coming from a controlled source that is guaranteed to be
                    # immutable.
                    bin_ = get_bin(kernel_build_env)
                else:
                    key = get_key(kernel_build_env)
                    bin_ = lookup_cache(kernel_build_env, key)
        else:
            bin_ = lookup_cache(kernel_build_env, key, enter_cm=True)

        return (bin_, kernel_build_env._to_spec())

    def install(self, kmod_params=None):
        """
        Install and load the module on the target.

        :param kmod_params: Parameters to pass to the module via ``insmod``.
            Non-string iterable values will be turned into a comma-separated
            string following the ``module_param_array()`` kernel API syntax.
        :type kmod_params: dict(str, object) or None
        """
        target = self.target

        def target_mktemp():
            return target.execute(
                f'mktemp -p {quote(target.working_directory)}'
            ).strip()

        @contextlib.contextmanager
        def kmod_cm():
            content = self._compile()
            with tempfile.NamedTemporaryFile('wb', suffix='.ko') as f:
                f.write(content)
                f.flush()

                target_temp = Path(target_mktemp())
                host_temp = Path(f.name)
                try:
                    target.push(str(host_temp), str(target_temp))
                    yield target_temp
                finally:
                    target.remove(str(target_temp))

        return self._install(kmod_cm(), kmod_params=kmod_params)

    def _install(self, kmod_cm, kmod_params):
        # Avoid circular import
        from lisa.trace import DmesgCollector

        def make_str(x):
            if isinstance(x, str):
                return x
            elif isinstance(x, collections.abc.Iterable):
                return ','.join(map(make_str, x))
            else:
                return str(x)

        def log_dmesg(coll, log):
            if coll:
                name = self.mod_name
                dmesg_entries = [
                    entry
                    for entry in coll.entries
                    if entry.msg.startswith(name)
                ]
                if dmesg_entries:
                    sep = '\n    '
                    dmesg = sep.join(map(str, dmesg_entries))
                    log(f'{name} kernel module dmesg output:{sep}{dmesg}')


        logger = self.logger
        target = self.target

        kmod_params = kmod_params or {}
        params = ' '.join(
            f'{quote(k)}={quote(make_str(v))}'
            for k, v in sorted(
                kmod_params.items(),
                key=itemgetter(0),
            )
        )

        try:
            self.uninstall()
        except Exception:
            pass

        with kmod_cm as ko_path, tempfile.NamedTemporaryFile() as dmesg_out:
            dmesg_coll = ignore_exceps(
                Exception,
                DmesgCollector(target, output_path=dmesg_out.name),
                lambda when, cm, excep: logger.error(f'Encountered exceptions while {when}ing dmesg collector: {excep}')
            )

            try:
                with dmesg_coll as dmesg_coll:
                    target.execute(f'{quote(target.busybox)} insmod {quote(str(ko_path))} {params}', as_root=True)

            except Exception as e:
                log_dmesg(dmesg_coll, logger.error)

                if isinstance(e, subprocess.CalledProcessError) and e.returncode == errno.EPROTO:
                    raise KmodVersionError('In-tree module version does not match what LISA expects. If the module was pre-installed on the target, please contact the 3rd party that shared this setup to you as they took responsibility for maintaining it. This setup is available but unsupported (see online documenation)')
                else:
                    raise
            else:
                log_dmesg(dmesg_coll, logger.debug)

    def uninstall(self):
        """
        Unload the module from the target.
        """
        mod = quote(self.mod_name)
        execute = lambda cmd: self.target.execute(cmd, as_root=True)

        try:
            execute(f'rmmod {mod}')
        except TargetStableError:
            execute(f'rmmod -f {mod}')

    @destroyablecontextmanager
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
        except ContextManagerExit:
            self.uninstall()


class FtraceDynamicKmod(DynamicKmod):
    """
    Dynamic module providing some custom ftrace events.
    """
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

        events = set(
            m.group(1)
            for m in map(
                parse,
                self._get_symbols('_ftrace_events')
            )
            if m
        )

        # Ensure that the possible_events() implementation is indeed a superset
        # of the events actually defined.
        assert set(self.possible_events) >= events

        return sorted(events)

    @property
    @memoized
    def possible_events(self):
        """
        Ftrace events possibly defined in that module.

        Note that this is based on crude source code analysis so it's expected
        to be a superset of the actually defined events.
        """
        decomment = re.compile(rb'^.*//.*?$|/\*.*?\*/', flags=re.MULTILINE)

        # We match trace_lisa__XXX tokens, as these are the events that are
        # actually used inside the module.
        find_event = re.compile(rb'\btrace_(lisa__.*?)\b')
        def find_events(code):
            code = decomment.sub(b'', code)
            return map(bytes.decode, find_event.findall(code))

        return sorted({
            possible_event
            for code in self.src.code_files.values()
            for possible_event in find_events(code)
        })


class _LISADynamicKmodSrc(KmodSrc):
    _RUST_SPEC = dict(
        version='1.83.0',
        components=[
            # rust-src for -Zbuild-std
            'rust-src',
        ]
    )


class LISADynamicKmod(FtraceDynamicKmod):
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
        mod_name = 'lisa'

        extra = {}

        path = Path(ASSETS_PATH) / 'kmodules' / 'lisa'
        btf_path = '/sys/kernel/btf/vmlinux'

        with tempfile.NamedTemporaryFile() as f:
            try:
                # We do not attempt to cache the BTF blob beyond the lifetime
                # of the target object. Any mistake in the caching key would
                # result in incredibly hard-to-debug issues due to mismatching
                # memory layout.
                target.cached_pull(btf_path, f.name, via_temp=True, as_root=True)
            except FileNotFoundError:
                raise FileNotFoundError(f'Could not find {btf_path} on the target. Ensure you compiled your kernel using CONFIG_DEBUG_INFO=y CONFIG_DEBUG_INFO_BTF=y CONFIG_DEBUG_INFO_REDUCED=n')

            with open(f.name, 'rb') as f:
                btf = f.read()

        extra['vmlinux'] = btf

        try:
            # Ensure addresses are real, since we will use them in a linker script
            with target.batch_revertable_write_value((
                dict(path='/proc/sys/kernel/kptr_restrict', value='0'),
                dict(path='/proc/sys/kernel/perf_event_paranoid', value='-1'),
            )):
                kallsyms = target.read_value('/proc/kallsyms')
        except TargetStableError:
            kallsyms = []
        else:
            kallsyms = parse_kallsyms(kallsyms)

        def sym_mod(module):
            if module:
                return f'\t[{module}]'
            else:
                return ''

        # Sort and filter kallsyms so that we have a stable content usable as a
        # cache key
        kallsyms = '\n'.join(
            f'{addr:x}\t{symtype}\t{name}{sym_mod(module)}'
            for (addr, name, symtype, module) in kallsyms
            # If the symbol is part of a module, it must be ignored, as some
            # symbols are duplicated in each module (e.g. __this_module), and
            # would conflict with the ones defined for the LISA module.
            if not module
        ) + '\n'
        kallsyms = kallsyms.encode('utf-8')
        extra['kallsyms'] = kallsyms

        logger = cls.get_logger()
        if logger.isEnabledFor(logging.DEBUG):
            extra_checksum = ', '.join(
                f'{name}={checksum(io.BytesIO(content), method="md5")}'
                for name, content in sorted(extra.items())
            )
            logger.debug(f'Variable sources checksum of the {cls.__qualname__} module: {extra_checksum}')

        src = _LISADynamicKmodSrc.from_path(path, extra=extra, name=mod_name)
        return cls(
            target=target,
            src=src,
            **kwargs,
        )

    def _event_features(self, events):
        all_events = self.defined_events
        return set(
            f'event__{event}'
            for pattern in events
            for event in fnmatch.filter(all_events, pattern)
        )

    def install(self, kmod_params=None):

        target = self.target
        logger = self.logger
        busybox = quote(target.busybox)

        def guess_kmod_path():
            modules_path_base = '/lib/modules'
            modules_version = target.kernel_version.release

            if target.os == 'android':
                modules_path_base = f'/vendor_dlkm{modules_path_base}'
                # Hack for GKI modules where the path might not match the kernel's
                # uname -r
                try:
                    modules_version = Path(target.execute(
                        f"{busybox} find {modules_path_base} -maxdepth 1 -mindepth 1 | {busybox} head -1"
                    ).strip()).name
                except subprocess.CalledProcessError:
                    pass

            base_path = f"{modules_path_base}/{modules_version}"
            return (base_path, f"{self.mod_name}.ko")


        kmod_params = kmod_params or {}
        kmod_params['version'] = self.src.checksum

        base_path, kmod_filename = guess_kmod_path()
        logger.debug(f'Looking for pre-installed {kmod_filename} module in {base_path}')

        super_ = super()
        def preinstalled_unsuitable(excep=None):
            if excep is not None:
                logger.debug(f'Pre-installed {kmod_filename} is unsuitable, recompiling: {excep.__class__.__qualname__}: {excep}')
            return super_.install(kmod_params=kmod_params)

        try:
            kmod_path = target.execute(
                f"{busybox} find {base_path} -name {quote(kmod_filename)}"
            ).strip()
        except subprocess.CalledProcessError:
            # If find fails, this means base_path does not even exist on the
            # target, so we just install the module
            return preinstalled_unsuitable()
        else:
            kmod_path = kmod_path.strip()
            if len((kmod_paths := kmod_path.splitlines())) > 1:
                return preinstalled_unsuitable(ValueError(f'Multiple paths found for {kmod_filename}: {", ".join(kmod_paths)}'))
            else:
                # We found an installed module that could maybe be suitable, so
                # we try to load it.
                try:
                    return self._install(nullcontext(kmod_path), kmod_params=kmod_params)
                except (subprocess.CalledProcessError, KmodVersionError) as e:
                    # Turns out to not be suitable, so we build our own
                    return preinstalled_unsuitable(e)
                else:
                    logger.warning(f'Loaded "{self.mod_name}" module from pre-installed location: {kmod_path}. This implies that the module was compiled by a 3rd party, which is available but unsupported. If you experience issues related to module version mismatch in the future, please contact them for updating the module. This may break at any time, without notice, and regardless of the general backward compatibility policy of LISA.')
                    return None


    def install(self, features=None, **kwargs):
        """
        Install and load the module on the target.

        :param features: Features to enable and associated parameters.
            Top-level in the dict is feature names, nested dict is for parameters.
        :type features: dict(str, dict(str, object)) or None
        """
        features = features or {}
        params = dict(
            features=sorted(features.keys()),
            **{
                f'{feature}___{name}': value
                for feature, params in features.items()
                for name, value in (params or {}).items()
            }
        )

        return super().install(kmod_params=params, **kwargs)


# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
