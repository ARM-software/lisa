LISA provides a Vagrant recipe which allows to automate the generation of a
VirtualBox based virtual machine pre-configured to run LISA. To generate
and use such a virtual machine you need:

- VirtualBox intalled in your machine, you can download the VM installer for
  your specific system from this page: https://www.virtualbox.org/wiki/Downloads

- Vagrant installed in your machine, you can download the installer for your
  specific system from this page: https://www.vagrantup.com/downloads.html
```bash
$ wget https://releases.hashicorp.com/vagrant/1.8.1/vagrant_1.8.1_x86_64.deb
$ sudo dpkg -i ./vagrant_1.8.1_x86_64.deb
```
Once these two components are available in your machine, to install LISA you
need to:

- clone the LISA repository in a local folder
```bash
# Clone the master LISA repository
$ git clone https://github.com/ARM-software/lisa.git
```
- create and start a Vagrant/VirtualBox VM
```bash
# Enter the LISA source tree
$ cd lisa
# Install LISA and its dependencies within the virtual machine
$ vagrant up
```	
This last command builds and execute the VM according to the description
provided by the Vagrant file available in the root folder of the LISA
source tree. The first time you run this command it will take some time
to download the based Ubuntu image and to install the required LISA
dependencies. The actual time depends on the speed of your internet
connection, the download size is rought 

When the installation complete you will get a prompt from the LISA shell
which is running withint the VM you just built. This VM shell can be
accessed from another terminal using this command
```bash
# from within the LISA root folder...
$ vagrant ssh
```
Once you exit all the LISA shell the VM is automatically stopped by vagrant.
The next time you run the "up" command the VM will be started again and you
will get a LISA shell.

From within the LISA shell you can start the IPython Notebooks server
by following the instructions in the following section.