This note assumes installation from scratch on a freshly installed
Ubuntu 16.04 system. If you have a different system (e.g. OSX or an
older version of Ubuntu) you can refer to the [Virtual
Machine based installation](https://github.com/ARM-software/lisa/wiki/Installation#virtual-machine-based-installation).

## Contents

1. [Standard Installation](https://github.com/ARM-software/lisa/wiki/Installation#standard-installation)
2. [Virtual Machine based installation](https://github.com/ARM-software/lisa/wiki/Installation#virtual-machine-based-installation)

## Standard Installation

### Required dependencies

	# Install common build related tools
	$ sudo apt-get install build-essential autoconf automake libtool pkg-config

	# Install additional tools required for some notebooks and tests
	$ sudo apt-get install trace-cmd sshpass kernelshark

	# Install optional tools required for some notebooks and tests
	$ sudo apt-get install nmap net-tools tree

	# Install required libraries packages
	$ sudo apt-get install libfreetype6-dev libpng12-dev

	# Install the Python package manager
	$ sudo apt-get install python-pip python-dev

	# Update the Python package manager (some distribution may have an outdated version)
	$ pip install --upgrade pip

	# Install (upgrade) required python packages
	$ sudo pip install --upgrade matplotlib numpy nose

	# Install (upgrade) required Python libraries
	$ sudo pip install --upgrade Cython trappy bart-py devlib psutil wrapt

	# Some specific notebooks may also require scipy
	$ sudo pip install --upgrade scipy

*NOTE:* TRAPpy and BART depend on `ipython` and `jupyter`. Some IPython
Notebooks examples are written using the notebooks JSON nbformat version 4,
which might not be supported by the IPython version installed by `apt-get`.
It is suggested to remove apt-get installed IPython and install it
using `pip`, which will provides the most updated version:

	# Remove (eventually) already installed versions
	$ sudo apt-get remove ipython ipython-notebook
	# Install most update version of the notebook
	$ sudo pip install ipython jupyter

### Installing Jupyter NBextensions

The Jupyter notebook server installed in the previous step is just a basic version.
Although it's just enough to open and use all the notebooks provided by LISA, if you
want to take maximum advantages of the Notebooks a standard set of extensions are
provided as a dependency repository.

To install the extensions you should follow the instructions on that official websiste:

   [[https://github.com/ipython-contrib/jupyter_contrib_nbextensions]]

The requires steps should be:

        $ pip install jupyter_contrib_nbextensions
        $ jupyter contrib nbextension install --user

### Clone the repository

The code of the LISA toolkit with all the supported tests and Notebooks can be
cloned from the official GitHub repository with this command:

        $ git clone https://github.com/ARM-software/lisa.git

### Clone the submodules

LISA depends on `bart`, `devlib` and `TRAPpy`.

        $ source init_env
        $ lisa-update submodules

## Virtual Machine based installation

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
This last command builds and executes the VM according to the description
provided by the Vagrant file available in the root folder of the LISA
source tree. The first time you run this command it will take some time
to download the based Ubuntu image and to install the required LISA
dependencies. The actual time depends on the speed of your internet
connection.

### Enable USB Controller

To be able to access devices connected through the USB, it is necessary to
enable the USB controller for the VM in `VirtualBox`. The following steps
explain how to do it:

- Halt the vagrant VM by running

```bash
vagrant halt
```

- Open the `VirtualBox` VM Manager

- Select the `lisa_default_*` virtual machine and click **Settings**

- Select the **USB** tab and enable the USB controller as shown in the screen-shot

[[images/vbox_enable_usb.png]]

- Add a **USB filter** for each of the devices that should be available in the VM as
shown in the screen-shot below

[[images/vbox_add_usb_filter.png]]

- Finally click **OK** to save the settings

It is now possible to start the VM by running:

```bash
vagrant up
```

### Run LISA in vagrant

When the installation complete you will get a prompt from the LISA shell
which is running within the VM you just built. This VM shell can be
accessed from another terminal using this command
```bash
# from within the LISA root folder...
$ vagrant ssh
```
Once you exit all the LISA shell the VM is automatically stopped by vagrant.
The next time you run the "up" command the VM will be started again and you
will get a LISA shell.

From within the LISA shell you can start the IPython Notebooks server
by following the instructions [here](https://github.com/ARM-software/lisa/wiki/Quickstart-Tutorial).
