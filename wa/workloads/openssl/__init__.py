#    Copyright 2018 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

from wa import Workload, Parameter, TargetError, WorkloadError, Executable, Alias
from wa.utils.exec_control import once


BLOCK_SIZES = [16, 64, 256, 1024, 8192, 16384]

ECD = ['secp160r1', 'nistp192', 'nistp224', 'nistp256', 'nistp384', 'nistp521',
       'nistk163', 'nistk233', 'nistk283', 'nistk409', 'nistk571', 'nistb163',
       'nistb233', 'nistb283', 'nistb409', 'nistb571', 'curve25519']

CIPHER_PKI = ['rsa', 'dsa', 'ecdh', 'ecdsa']
EVP_NEW = ['aes-128-cbc', 'aes-192-cbc', 'aes-256-cbc', 'aes-128-gcm', 'aes-192-gcm',
           'aes-256-gcm', 'sha1', 'sha256', 'sha384', 'sha512']


class Openssl(Workload):

    name = 'openssl'

    description = '''
    Benchmark Openssl algorithms using Openssl's speed command.

    The command tests how long it takes to perfrom typical SSL operations using
    a range of supported algorithms and ciphers.

    By defalt, this workload will use openssl installed on the target, however
    it is possible to provide an alternative binary as a workload resource.

    '''

    parameters = [
        Parameter('algorithm', default='aes-256-cbc',
                  allowed_values=EVP_NEW + CIPHER_PKI,
                  description='''
                  Algorithm to benchmark.
                  '''),
        Parameter('threads', kind=int, default=1,
                  description='''
                  The number of threads to use
                  '''),
        Parameter('use_system_binary', kind=bool, default=True,
                  description='''
                  If ``True``, the system Openssl binary will be used.
                  Otherwise, use the binary provided in the workload
                  resources.
                  '''),
    ]

    aliases = [Alias('ossl-' + algo, algorithm=algo)
               for algo in EVP_NEW + CIPHER_PKI]

    @once
    def initialize(self, context):
        if self.use_system_binary:
            try:
                cmd = '{0} md5sum < $({0} which openssl)'
                output = self.target.execute(cmd.format(self.target.busybox))
                md5hash = output.split()[0]
                version = self.target.execute('openssl version').strip()
                context.update_metadata('hashes', 'openssl', md5hash)
                context.update_metadata('versions', 'openssl', version)
            except TargetError:
                msg = 'Openssl does not appear to be installed on target.'
                raise WorkloadError(msg)
            Openssl.target_exe = 'openssl'
        else:
            resource = Executable(self, self.target.abi, 'openssl')
            host_exe = context.get_resource(resource)
            Openssl.target_exe = self.target.install(host_exe)

    def setup(self, context):
        self.output = None
        if self.algorithm in EVP_NEW:
            cmd_template = '{} speed -mr -multi {} -evp {}'
        else:
            cmd_template = '{} speed -mr -multi {} {}'
        self.command = cmd_template.format(self.target_exe, self.threads, self.algorithm)

    def run(self, context):
        self.output = self.target.execute(self.command)

    def extract_results(self, context):
        if not self.output:
            return

        outfile = os.path.join(context.output_directory, 'openssl.output')
        with open(outfile, 'w') as wfh:
            wfh.write(self.output)
        context.add_artifact('openssl-output', outfile, 'raw', 'openssl\'s stdout')

    def update_output(self, context):
        if not self.output:
            return

        for line in self.output.split('\n'):
            line = line.strip()

            if not line.startswith('+F'):
                continue

            parts = line.split(':')
            if parts[0] == '+F':  # evp ciphers
                for bs, value in zip(BLOCK_SIZES, list(map(float, parts[3:]))):
                    value = value / 2**20  # to MB
                    context.add_metric('score', value, 'MB/s',
                                       classifiers={'block_size': bs})
            elif parts[0] in ['+F2', '+F3']:  # rsa, dsa
                key_len = int(parts[2])
                sign = float(parts[3])
                verify = float(parts[4])
                context.add_metric('sign', sign, 'seconds',
                                   classifiers={'key_length': key_len})
                context.add_metric('verify', verify, 'seconds',
                                   classifiers={'key_length': key_len})
            elif parts[0] == '+F4':  # ecdsa
                ec_idx = int(parts[1])
                key_len = int(parts[2])
                sign = float(parts[3])
                verify = float(parts[4])
                context.add_metric('sign', sign, 'seconds',
                                   classifiers={'key_length': key_len,
                                                'curve': ECD[ec_idx]})
                context.add_metric('verify', verify, 'seconds',
                                   classifiers={'key_length': key_len,
                                                'curve': ECD[ec_idx]})
            elif parts[0] == '+F5':  # ecdh
                ec_idx = int(parts[1])
                key_len = int(parts[2])
                op_time = float(parts[3])
                ops_per_sec = float(parts[4])
                context.add_metric('op', op_time, 'seconds',
                                   classifiers={'key_length': key_len,
                                                'curve': ECD[ec_idx]})
                context.add_metric('ops_per_sec', ops_per_sec, 'Hz',
                                   classifiers={'key_length': key_len,
                                                'curve': ECD[ec_idx]})
            else:
                self.logger.warning('Unexpected result: "{}"'.format(line))

    @once
    def finalize(self, context):
        if not self.use_system_binary and self.uninstall:
            self.target.uninstall('openssl')
