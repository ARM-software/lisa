import os
import sys
from subprocess import Popen, PIPE

def get_commit():
    p = Popen(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__),
              stdout=PIPE, stderr=PIPE)
    std, _ = p.communicate()
    p.wait()
    if p.returncode:
        return None
    if sys.version_info[0] == 3:
        return std[:8].decode(sys.stdout.encoding)
    else:
        return std[:8]
