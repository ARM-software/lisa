This note assumes installation from scratch on a freshly installed
Ubuntu 16.04 system. If you have a different system (e.g. OSX or an
older version of Ubuntu, you can probably be interesting into a virtual
machine based installation, which is described in a following section.

## Required dependencies

	# Install common build related tools
	$ sudo apt-get install build-essential autoconf automake libtool pkg-config

	# Install additional tools required for some notebooks and tests
	$ sudo apt-get install trace-cmd sshpass kernelshark

	# Install optional tools required for some notebooks and tests
	$ sudo apt-get install nmap net-tools tree

	# Install required python packages
	$ sudo apt-get install python-matplotlib python-numpy libfreetype6-dev libpng12-dev python-nose

	# Install the Python package manager
	$ sudo apt-get install python-pip python-dev

	# Install (upgrade) required Python libraries
	$ sudo pip install --upgrade trappy bart-py devlib

*NOTE:* TRAPpy and BART depend on `ipython` and `jupyter`. Some IPython
Notebooks examples are written using the notebooks JSON nbformat version 4,
which might not be supported by the IPython version installed by `apt-get`.
It is suggested to remove apt-get installed IPython and install it
using `pip`, which will provides the most updated version:

	# Remove (eventually) already installed versions
	$ sudo apt-get remove ipython ipython-notebook
	# Install most update version of the notebook
	$ sudo pip install ipython jupyter

## Clone the repository

The code of the LISA toolkit with all the supported tests and Notebooks can be
cloned from the official GitHub repository with this command:

        $ git clone https://github.com/ARM-software/lisa.git