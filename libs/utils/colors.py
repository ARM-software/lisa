
import sys

class TestColors:

    level = {
        'failed'    : '\033[0;31m', # Red
        'good'      : '\033[0;32m', # Green
        'warning'   : '\033[0;33m', # Yellow
        'passed'    : '\033[0;34m', # Blue
        'purple'    : '\033[0;35m', # Purple
        'endc'      : '\033[0m'     # End color
    }

    @staticmethod
    def rate(val, positive_is_good=True):
        str_val = "{:9.2f}%".format(val)

        if not sys.stdout.isatty():
            return str_val

        if not positive_is_good:
            val = -val

        if val < -10:
            str_color = TestColors.level['failed']
        elif val < 0:
            str_color = TestColors.level['warning']
        elif val < 10:
            str_color = TestColors.level['passed']
        else:
            str_color = TestColors.level['good']

        return str_color + str_val + TestColors.level['endc']

#vim :set tabstop=4 shiftwidth=4 expandtab
