import unittest

from louie.robustapply import robust_apply


def no_argument():
    pass


def one_argument(blah):
    pass


def two_arguments(blah, other):
    pass


class TestRobustApply(unittest.TestCase):
    
    def test_01(self):
        robust_apply(no_argument, no_argument)
        
    def test_02(self):
        self.assertRaises(TypeError, robust_apply, no_argument, no_argument,
                          'this' )
        
    def test_03(self):
        self.assertRaises(TypeError, robust_apply, one_argument, one_argument)
        
    def test_04(self):
        """Raise error on duplication of a particular argument"""
        self.assertRaises(TypeError, robust_apply, one_argument, one_argument,
                          'this', blah='that')

