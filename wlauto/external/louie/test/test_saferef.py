import unittest

from louie.saferef import safe_ref


class _Sample1(object):
    def x(self):
        pass

    
def _sample2(obj):
    pass


class _Sample3(object):
    def __call__(self, obj):
        pass

    
class TestSaferef(unittest.TestCase):

    # XXX: The original tests had a test for closure, and it had an
    # off-by-one problem, perhaps due to scope issues.  It has been
    # removed from this test suite.
    
    def setUp(self):
        ts = []
        ss = []
        for x in xrange(5000):
            t = _Sample1()
            ts.append(t)
            s = safe_ref(t.x, self._closure)
            ss.append(s)
        ts.append(_sample2)
        ss.append(safe_ref(_sample2, self._closure))
        for x in xrange(30):
            t = _Sample3()
            ts.append(t)
            s = safe_ref(t, self._closure)
            ss.append(s)
        self.ts = ts
        self.ss = ss
        self.closure_count = 0
        
    def tearDown(self):
        if hasattr(self, 'ts'):
            del self.ts
        if hasattr(self, 'ss'):
            del self.ss
        
    def test_In(self):
        """Test the `in` operator for safe references (cmp)"""
        for t in self.ts[:50]:
            assert safe_ref(t.x) in self.ss
            
    def test_Valid(self):
        """Test that the references are valid (return instance methods)"""
        for s in self.ss:
            assert s()
            
    def test_ShortCircuit(self):
        """Test that creation short-circuits to reuse existing references"""
        sd = {}
        for s in self.ss:
            sd[s] = 1
        for t in self.ts:
            if hasattr(t, 'x'):
                assert sd.has_key(safe_ref(t.x))
            else:
                assert sd.has_key(safe_ref(t))
                
    def test_Representation(self):
        """Test that the reference object's representation works

        XXX Doesn't currently check the results, just that no error
            is raised
        """
        repr(self.ss[-1])
        
    def _closure(self, ref):
        """Dumb utility mechanism to increment deletion counter"""
        self.closure_count += 1

