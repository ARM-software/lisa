# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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

from unittest import TestCase

import pandas as pd

import lisa.datautils as du

class DfCheck(TestCase):
    def test_df_split_signals(self):
        index = list(map(float, range(1, 6)))
        cols = ["foo", "bar"]
        ncols = len(cols)

        data = [(i, i % 2) for i in range(len(index))]

        df = pd.DataFrame(index=index, data=data, columns=cols)

        for ident, subdf in du.df_split_signals(df, ["bar"]):
            if ident["bar"] == 0:
                assert len(subdf) == 3
            else:
                assert len(subdf) == 2
