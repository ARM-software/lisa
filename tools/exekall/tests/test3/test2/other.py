from tests.testcode import AssetBundle, ResultBundle
from logging import Logger

def mytest(assets:AssetBundle, log:Logger=None, margin=42) -> ResultBundle:
    #  print('testing mytest() with margin={x}...'.format(x=margin))
    print('testing mytest() with storage={x}...'.format(x=assets.storage))
    return ResultBundle('hello mytest', None)


class Temp: pass
def f(t:Temp) -> ResultBundle:
    return None

class A:
    @classmethod
    def f(cls) -> Temp:
        print(cls)
        return Temp()

class B(A):
    pass

