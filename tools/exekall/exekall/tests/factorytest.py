#! /usr/bin/env python3

import numbers
from logging import Logger
import engine
import utils
from lisa_glue import ArtifactStorage, TargetConf
from tests.testcode import ResultBundle

class Bar:
    pass
    @classmethod
    def from_x(cls) -> 'Bar':
        return cls()

    #  def bar_to_res_base(self) -> ResultBundle:
        #  return ResultBundle('from bar', self)

class SubBar(Bar):
    def bar_to_res(self) -> ResultBundle:
        return ResultBundle('from subbar', self)

class SubBar2(SubBar):
    def bar_to_res2(self) -> ResultBundle:
        return ResultBundle('from subar2', self)
