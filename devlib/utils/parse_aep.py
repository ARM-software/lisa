#!/usr/bin/env python
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

#    Copyright 2018 Linaro Limited
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

import getopt
import logging
import signal
import sys

logger = logging.getLogger('aep-parser')

# pylint: disable=attribute-defined-outside-init
class AepParser(object):
    prepared = False

    @staticmethod
    def topology_from_data(array, topo):
    # Extract topology information for the data file
    # The header of a data file looks like this ('#' included):
    # configuration: <file path>
    # config_name: <file name>
    # trigger: 0.400000V (hyst 0.200000V) 0.000000W (hyst 0.200000W) 400us
    # date: Fri, 10 Jun 2016 11:25:07 +0200
    # host: <host name>
    #
    # CHN_0	    Pretty_name_0   PARENT_0    Color0  Class0
    # CHN_1	    Pretty_name_1   PARENT_1    Color1  Class1
    # CHN_2	    Pretty_name_2   PARENT_2    Color2  Class2
    # CHN_3	    Pretty_name_3   PARENT_3    Color3  Class3
    # ..
    # CHN_N	    Pretty_name_N   PARENT_N    ColorN  ClassN
    #

        info = {}

        if len(array) == 6:
            info['name'] = array[1]
            info['parent'] = array[3]
            info['pretty'] = array[2]
            # add an entry for both name and pretty name in order to not parse
            # the whole dict when looking for a parent and the parent of parent
            topo[array[1]] = info
            topo[array[2]] = info
        return topo

    @staticmethod
    def create_virtual(topo, label, hide, duplicate):
    # Create a list of virtual power domain that are the sum of others
    # A virtual domain is the parent of several channels but is not sampled by a
    # channel
    # This can be useful if a power domain is supplied by 2 power rails
        virtual = {}

        # Create an entry for each virtual parent
        for supply in topo.keys():
            index = topo[supply]['index']
            # Don't care of hidden columns
            if hide[index]:
                continue

            # Parent is in the topology
            parent = topo[supply]['parent']
            if parent in topo:
                continue

            if parent not in virtual:
                virtual[parent] = {supply : index}

            virtual[parent][supply] = index

        # Remove parent with 1 child as they don't give more information than their
        # child
        for supply in list(virtual.keys()):
            if len(virtual[supply]) == 1:
                del virtual[supply]

        for supply in list(virtual.keys()):
            # Add label, hide and duplicate columns for virtual domains
            hide.append(0)
            duplicate.append(1)
            label.append(supply)

        return virtual

    @staticmethod
    def get_label(array):
    # Get the label of each column
    # Remove unit '(X)' from the end of the label
        label = [""]*len(array)
        unit = [""]*len(array)

        label[0] = array[0]
        unit[0] = "(S)"
        for i in range(1, len(array)):
            label[i] = array[i][:-3]
            unit[i] = array[i][-3:]

        return label, unit

    @staticmethod
    def filter_column(label, unit, topo):
    # Filter columns
    # We don't parse Volt and Amper columns: put in hide list
    # We don't add in Total a column that is the child of another one: put in duplicate list

        # By default we hide all columns
        hide = [1] * len(label)
        # By default we assume that there is no child
        duplicate = [0] * len(label)

        for i in range(len(label)):  # pylint: disable=consider-using-enumerate
            # We only care about time and Watt
            if label[i] == 'time':
                hide[i] = 0
                continue

            if '(W)' not in unit[i]:
                continue

            hide[i] = 0

            #label is pretty name
            pretty = label[i]

            # We don't add a power domain that is already accounted by its parent
            if topo[pretty]['parent'] in topo:
                duplicate[i] = 1

            # Set index, that will be used by virtual domain
            topo[topo[pretty]['name']]['index'] = i

            # remove pretty element that is useless now
            del topo[pretty]

        return hide, duplicate

    @staticmethod
    def parse_text(array, hide):
        data = [0]*len(array)
        for i in range(len(array)):  # pylint: disable=consider-using-enumerate
            if hide[i]:
                continue

            try:
                data[i] = int(float(array[i])*1000000)
            except ValueError:
                continue

        return data

    @staticmethod
    def add_virtual_data(data, virtual):
        # write virtual domain
        for parent in virtual.keys():
            power = 0
            for child in list(virtual[parent].values()):
                try:
                    power += data[child]
                except IndexError:
                    continue
            data.append(power)

        return data

    @staticmethod
    def delta_nrj(array, delta, minimu, maximum, hide):
    # Compute the energy consumed in this time slice and add it
    # delta[0] is used to save the last time stamp

        if delta[0] < 0:
            delta[0] = array[0]

        time = array[0] - delta[0]
        if time <= 0:
            return delta

        for i in range(len(array)):  # pylint: disable=consider-using-enumerate
            if hide[i]:
                continue

            try:
                data = array[i]
            except ValueError:
                continue

            if data < minimu[i]:
                minimu[i] = data
            if data > maximum[i]:
                maximum[i] = data
            delta[i] += time * data

        # save last time stamp
        delta[0] = array[0]

        return delta

    def output_label(self, label, hide):
        self.fo.write(label[0] + "(uS)")
        for i in range(1, len(label)):
            if hide[i]:
                continue
            self.fo.write(" " + label[i] + "(uW)")

        self.fo.write("\n")

    def output_power(self, array, hide):
        #skip partial line. Most probably the last one
        if len(array) < len(hide):
            return

        # write not hidden colums
        self.fo.write(str(array[0]))
        for i in range(1, len(array)):
            if hide[i]:
                continue

            self.fo.write(" "+str(array[i]))

        self.fo.write("\n")

    # pylint: disable-redefined-outer-name,
    def prepare(self, input_file, outfile, summaryfile):
        try:
            self.fi = open(input_file, "r")
        except IOError:
            logger.warning('Unable to open input file {}'.format(input_file))
            logger.warning('Usage: parse_arp.py -i <inputfile> [-o <outputfile>]')
            sys.exit(2)

        self.parse = True
        if outfile:
            try:
                self.fo = open(outfile, "w")
            except IOError:
                logger.warning('Unable to create {}'.format(outfile))
                self.parse = False
        else:
            self.parse = False

        self.summary = True
        if summaryfile:
            try:
                self.fs = open(summaryfile, "w")
            except IOError:
                logger.warning('Unable to create {}'.format(summaryfile))
                self.fs = sys.stdout
        else:
            self.fs = sys.stdout

        self.prepared = True

    def unprepare(self):
        if not self.prepared:
            # nothing has been prepared
            return

        self.fi.close()

        if self.parse:
            self.fo.close()

        self.prepared = False

    # pylint: disable=too-many-branches,too-many-statements,redefined-outer-name,too-many-locals
    def parse_aep(self, start=0, length=-1):
    # Parse aep data and calculate the energy consumed
        begin = 0

        label_line = 1

        topo = {}

        lines = self.fi.readlines()

        for myline in lines:
            array = myline.split()

            if "#" in myline:
                # update power topology
                topo = self.topology_from_data(array, topo)
                continue

            if label_line:
                label_line = 0
                # 1st line not starting with # gives label of each column
                label, unit = self.get_label(array)
                # hide useless columns and detect channels that are children
                # of other channels
                hide, duplicate = self.filter_column(label, unit, topo)

                # Create virtual power domains
                virtual = self.create_virtual(topo, label, hide, duplicate)
                if self.parse:
                    self.output_label(label, hide)

                logger.debug('Topology : {}'.format(topo))
                logger.debug('Virtual power domain : {}'.format(virtual))
                logger.debug('Duplicated power domain : : {}'.format(duplicate))
                logger.debug('Name of columns : {}'.format(label))
                logger.debug('Hidden columns : {}'.format(hide))
                logger.debug('Unit of columns : {}'.format(unit))

                # Init arrays
                nrj = [0]*len(label)
                minimum = [100000000]*len(label)
                maximum = [0]*len(label)
                offset = [0]*len(label)

                continue

            # convert text to int and unit to micro-unit
            data = self.parse_text(array, hide)

            # get 1st time stamp
            if begin <= 0:
                begin = data[0]

            # skip data before start
            if (data[0]-begin) < start:
                continue

            # stop after length
            if length >= 0 and (data[0]-begin) > (start + length):
                continue

            # add virtual domains
            data = self.add_virtual_data(data, virtual)

            # extract power figures
            self.delta_nrj(data, nrj, minimum, maximum, hide)

            # write data into new file
            if self.parse:
                self.output_power(data, hide)

        # if there is no data just return
        if label_line or len(nrj) == 1:
            raise ValueError('No data found in the data file. Please check the Arm Energy Probe')

        # display energy consumption of each channel and total energy consumption
        total = 0
        results_table = {}
        for i in range(1, len(nrj)):
            if hide[i]:
                continue

            nrj[i] -= offset[i] * nrj[0]

            total_nrj = nrj[i]/1000000000000.0
            duration = (maximum[0]-minimum[0])/1000000.0
            channel_name = label[i]
            average_power = total_nrj/duration

            total = nrj[i]/1000000000000.0
            duration = (maximum[0]-minimum[0])/1000000.0
            min_power = minimum[i]/1000000.0
            max_power = maximum[i]/1000000.0
            output = "Total nrj: %8.3f J for %s -- duration %8.3f sec -- min %8.3f W -- max %8.3f W\n"
            self.fs.write(output.format(total, label[i], duration, min_power, max_power))

            # store each AEP channel info  except Platform in the results table
            results_table[channel_name] = total_nrj, average_power

            if minimum[i] < offset[i]:
                self.fs.write("!!! Min below offset\n")

            if duplicate[i]:
                continue

            total += nrj[i]

        output = "Total nrj: %8.3f J for Platform  -- duration %8.3f sec\n"
        self.fs.write(output.format(total/1000000000000.0, (maximum[0]-minimum[0])/1000000.0))

        total_nrj = total/1000000000000.0
        duration = (maximum[0]-minimum[0])/1000000.0
        average_power = total_nrj/duration

        # store AEP Platform channel info in the results table
        results_table["Platform"] = total_nrj, average_power

        return results_table

    # pylint: disable=too-many-branches,no-self-use,too-many-locals
    def topology_from_config(self, topofile):
        try:
            ft = open(topofile, "r")
        except IOError:
            logger.warning('Unable to open config file {}'.format(topofile))
            return
        lines = ft.readlines()

        topo = {}
        virtual = {}
        name = ""
        offset = 0
        index = 0
        #parse config file
        for myline in lines:
            if myline.startswith("#"):
                # skip comment
                continue

            if myline == "\n":
                # skip empty line
                continue

            if name == "":
                # 1st valid line is the config's name
                name = myline
                continue

            if not myline.startswith((' ', '\t')):
                # new device path
                offset = index
                continue

            # Get parameters of channel configuration
            items = myline.split()

            info = {}
            info['name'] = items[0]
            info['parent'] = items[9]
            info['pretty'] = items[8]
            info['index'] = int(items[2])+offset

            # Add channel
            topo[items[0]] = info

            # Increase index
            index += 1


        # Create an entry for each virtual parent
        # pylint: disable=consider-iterating-dictionary
        for supply in topo.keys():
            # Parent is in the topology
            parent = topo[supply]['parent']
            if parent in topo:
                continue

            if parent not in virtual:
                virtual[parent] = {supply : topo[supply]['index']}

            virtual[parent][supply] = topo[supply]['index']


        # Remove parent with 1 child as they don't give more information than their
        # child
        # pylint: disable=consider-iterating-dictionary
        for supply in list(virtual.keys()):
            if len(virtual[supply]) == 1:
                del virtual[supply]

        topo_list = ['']*(1+len(topo)+len(virtual))
        topo_list[0] = 'time'
        # pylint: disable=consider-iterating-dictionary
        for chnl in topo.keys():
            topo_list[topo[chnl]['index']] = chnl
        for chnl in virtual.keys():
            index += 1
            topo_list[index] = chnl

        ft.close()

        return topo_list

    def __del__(self):
        self.unprepare()

if __name__ == '__main__':

    # pylint: disable=unused-argument
    def handleSigTERM(signum, frame):
        sys.exit(2)

    signal.signal(signal.SIGTERM, handleSigTERM)
    signal.signal(signal.SIGINT, handleSigTERM)

    logger.setLevel(logging.WARN)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    in_file = ""
    out_file = ""
    figurefile = ""
    start = 0
    length = -1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:vo:s:l:t:")
    except getopt.GetoptError as err:
        print(str(err)) # will print something like "option -a not recognized"
        sys.exit(2)

    for o, a in opts:
        if o == "-i":
            in_file = a
        if o == "-v":
            logger.setLevel(logging.DEBUG)
        if o == "-o":
            parse = True
            out_file = a
        if o == "-s":
            start = int(float(a)*1000000)
        if o == "-l":
            length = int(float(a)*1000000)
        if o == "-t":
            topfile = a
            parser = AepParser()
            print(parser.topology_from_config(topfile))
            exit(0)

    parser = AepParser()
    parser.prepare(in_file, out_file, figurefile)
    parser.parse_aep(start, length)
