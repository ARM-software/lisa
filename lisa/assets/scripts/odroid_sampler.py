#!/usr/bin/python
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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

import logging
import getopt
import sys

# Sampler configuration
config = {
    'period'  : 0,
    'samples' : 0,
}

class OdroidSampler(object):

    sysfs_base = '/sys/devices/12c60000.i2c/i2c-4/i2c-dev/i2c-4/device'
    sysfs = {
        'arm' : sysfs_base + '/4-0040',
        'kfc' : sysfs_base + '/4-0045',
    }
    power = {
        'arm' : sysfs['arm'] + '/sensor_W',
        'kfc' : sysfs['kfc'] + '/sensor_W',
    }

    def __init__(self, samples, period):
        self.energy_proxy = {
            'arm' : 0,
            'kfc' : 0,
        }
        self.samples = samples
        self.period = period

    def sample(self, device):
        f = open(self.power[device], 'r')
        power = f.readline()
        self.energy_proxy[device] += power
        f.close()

    def averagePower(self, device):
        self.energy_proxy[device] = 0
        for i in range(0, self.samples):
            sample(self, device)
            sleep(self.period / 1e6)
        return self.energy_proxy[device] / self.samples


def parseOptions():
    global config

    logging.debug('Parsing options')

    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:s:", ["period=", "samples="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)
        usage()
        sys.exit(2)
    output = None
    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-p", "--period"):
            config['period'] = int(a)
        elif o in ("-s", "--samples"):
            config['samples'] = int(a)
        else:
            assert False, "unhandled option"

    logging.info('Sampler configured for {0:d} samples, evenry {1:.3f}ms'\
        .format(config['samples'], config['period']/1000))

def main():

    parseOptions()

    sampler = OdroidSampler(config['samples'], config['period'])

    avg_power = amples.averagePower(config['device'])
    logging.info('Average power: {0:f}'.format(avg_power))

logging.basicConfig(
    format='%(asctime)-9s %(levelname)-8s: %(message)s',
    level=logging.DEBUG,
    datefmt='%I:%M:%S')

if __name__ == "__main__":
        main()
