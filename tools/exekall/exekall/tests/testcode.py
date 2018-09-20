#! /usr/bin/env python3

import numbers
from logging import Logger
import engine
import utils
from lisa_glue import ArtifactStorage, TargetConf

class ResultBundle:
    def __init__(self, msg, asset):
        self.msg = msg
        self.asset = asset

    def __str__(self):
        return self.msg + ' on ' + str(self.asset)

    def __repr__(self):
        return str(self)

class Target:
    def __init__(self, conf:TargetConf):
        self.conf = conf

class AssetBundle:

    @utils.unique_type('margin', 'margin2')
    def __init__(self, target:Target, storage:ArtifactStorage, margin:'float'=42.0, margin2:'float'=44.0):
        self.storage = storage

        if target.conf.name == 'hikey960':
            raise ValueError('I am crashing on the 960, what a surprise !!!!')

        self.target = target
        print('Creating ' + self.__class__.__qualname__ + ', target:' + str(target.conf))
        self.asset_path = '/path/to/asset.dat'
        self.storage = storage
        self.margin = margin

        if margin == 3:
            raise ValueError('Cannot deal with that margin !!!')

    def __str__(self):
        return '<asset ' +type(self).__name__ + ' on target: '+str(self.target.conf)+' >'

    def __repr__(self):
        return str(self)

    def test_generic(self, storage:ArtifactStorage, log:Logger, margin=42) -> ResultBundle:
        print('test_generic testing ...')
        #  log.warn('test_generic testing ...')
        print('artifact storage: ' + str(storage))
        return ResultBundle('passed', self)

    def test_generic2(self, storage:ArtifactStorage, log:Logger, margin=42) -> ResultBundle:
        print('test_generic2 testing ...')
        #  log.warn('test_generic2 testing ...')
        return ResultBundle('passed', self)

    @staticmethod
    def test_generic3(asset:'AssetBundle', storage:ArtifactStorage, log:Logger, margin=42) -> ResultBundle:
        print('test_generic3 testing ...')
        #  log.warn('test_generic2 testing ...')
        return ResultBundle('passed', asset)

    @classmethod
    def test_generic_cls_meth(cls, asset:'AssetBundle', storage:ArtifactStorage, log:Logger, margin=42) -> ResultBundle:
        print('test_generic_cls_meth testing ...')
        #  log.warn('test_generic2 testing ...')
        return ResultBundle('passed', asset)

class GenericAssetBundle(AssetBundle):
    def test_special(self, log:Logger) -> ResultBundle:
        print('test_special testing ...')
        return ResultBundle('passed', self)

#  def create_result() -> ResultBundle:
    #  return ResultBundle('hello', None)

# Check the behavior when encountering cyclic dependencies
class Foo: pass
def introduce_cycle(res:ResultBundle) -> Foo:
    pass

def introduce_cycle2(foo:Foo) -> ResultBundle:
    pass

class MyClass:
    @staticmethod
    @utils.unique_type('margin')
    def custom_type_test(log:Logger, margin:'int'=3) -> ResultBundle:
        log.info('margin: {}, {}'.format(margin, type(margin).mro()))
        if margin == 2:
            raise ValueError('Cannot deal with that margin !!!')
        pass

def test_margin(log:Logger, margin:'int'=4) -> ResultBundle:
    log.info(margin)

#  def custom_type_test2(log:Logger, margin:'int') -> ResultBundle:
    #  log.info('margin: {}'.format(type(margin).mro()))
    #  pass

