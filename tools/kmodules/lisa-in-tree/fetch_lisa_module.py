#! /usr/bin/env python3

import shutil
import argparse
import subprocess

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="""
    Fetch the Lisa module from GitHub or a local copy and integrate it
    inside the kernel tree.

    Method 1 - using LISA_HOME.

    If the script finds the LISA_HOME environment variable (present after sourcing the Lisa shell)
    it will symlink the Lisa module sources from the local Lisa git checkout into the kernel tree.
    That way updating the local Lisa tree will automatically update the module sources in the kernel
    tree and the changes will be present after the next kernel rebuild.
    In this case, the script only needs to be run once.

    Method 2 - Sparse checkout.

    If the script doesn't find LISA_HOME (e.g. building on an external server without a Lisa checkout)
    the alternative mode is to directly clone the module sources into the tree.
                                     """)
    parser.add_argument('-g', '--git-ref', help='Lisa git reference to checkout')
    parser.add_argument('-f', '--force', action='store_true', help='Override the module checkout if it already exists')
    parser.add_argument('--module-kernel-path',
                        help='Path relative to the kernel tree root where the module will be stored')
    parser.add_argument('--git-remote', help='Git remote to pull the module from',
                        default='git@github.com:ARM-software/lisa.git')
    args = parser.parse_args()

    module_kernel_path = Path(args.module_kernel_path).resolve()

    # --git-ref passed, clone the sparse checkout
    if args.git_ref:
        module_git_path = module_kernel_path.parent / f"{module_kernel_path.name}-git"

        # module has already been cloned
        if module_git_path.exists():
            if not args.force:
                parser.error('Module checkout already exists, pass --force to override')
            shutil.rmtree(module_git_path)
            module_kernel_path.unlink()

        clone_cmd = [
            'git', 'clone', '--verbose', '--no-checkout', '--no-tags', '--filter=tree:0',
            args.git_remote, module_git_path
        ]
        subprocess.check_call(clone_cmd)
        subprocess.check_call(['git', '-C', module_git_path,
                               'sparse-checkout', 'set', 'lisa/_assets/kmodules/'])
        subprocess.check_call(['git', '-C', module_git_path, 'checkout', args.git_ref])

        module_src_kernel_git_path = module_git_path / 'lisa/_assets/kmodules/lisa'
        module_kernel_path.symlink_to(
            module_src_kernel_git_path.relative_to(module_kernel_path.parent)
        )
        return

    # --git-ref not passed, try linking from LISA_HOME
    try:
        import lisa._assets.kmodules.lisa as kmod_mod
    except ImportError:
        parser.error('Either lisa Python package needs to be importable or --git-ref needs to be used')
    else:
        module_src_lisa_path = Path(kmod_mod.__path__[0]).resolve()
        module_kernel_path.symlink_to(module_src_lisa_path)


main()
