import os
import shutil

from wa.framework.configuration.core import settings

# Have to disable this due to dynamic attributes
# pylint: disable=no-member

def init_user_directory(overwrite_existing=False):  # pylint: disable=R0914
    """
    Initialise a fresh user directory.
    """
    if os.path.exists(settings.user_directory):
        if not overwrite_existing:
            raise RuntimeError('Environment {} already exists.'.format(settings.user_directory))
        shutil.rmtree(settings.user_directory)

    os.makedirs(settings.user_directory)
    os.makedirs(settings.dependencies_directory)
    os.makedirs(settings.plugins_directory)

    # TODO: generate default config.yaml here

    if os.getenv('USER') == 'root':
        # If running with sudo on POSIX, change the ownership to the real user.
        real_user = os.getenv('SUDO_USER')
        if real_user:
            import pwd  # done here as module won't import on win32
            user_entry = pwd.getpwnam(real_user)
            uid, gid = user_entry.pw_uid, user_entry.pw_gid
            os.chown(settings.user_directory, uid, gid)
            # why, oh why isn't there a recusive=True option for os.chown?
            for root, dirs, files in os.walk(settings.user_directory):
                for d in dirs:
                    os.chown(os.path.join(root, d), uid, gid)
                for f in files:
                    os.chown(os.path.join(root, f), uid, gid)
