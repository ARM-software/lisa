# Introduction

The Behavioural Analysis and Regression Toolkit is based on [TRAPpy](https://github.com/ARM-software/trappy). The primary goal is to assert behaviours using the FTrace output from the kernel

## Target Audience
The framework is designed to cater to a wide range of audience. Aiding developers as well as automating
the testing of "difficult to test" behaviours.

### Kernel Developers

Making sure that the code that you are writing is doing the right thing.

### Performance Engineers

Plotting/Asserting performance behaviours between different revisions of the kernel

### Quality Assurance/Release Engineers
Verifying behaviours when different components/patches are integrated

# Installation

Clone the [BART]( https://github.com/ARM-software/bart) and [TRAPpy]( https://github.com/ARM-software/trappy) repos

    git clone git@github.com:ARM-software/bart.git
    git clone git@github.com:ARM-software/trappy.git

Add the directories to your PYTHONPATH

    export PYTHONPATH=$BASE_DIR/bart:$BASE_DIR/trappy:$PYTHONPATH

Install dependencies

    apt-get install ipython-notebook python-pandas

[IPython](http://ipython.org/notebook.html) notebook is a web based interactive python programming interface.
It is required if you plan to use interactive plotting in BART.

# Trace Analysis Language

BART also provides a generic Trace Analysis Language, which allows the user to construct complex relation statements on trace data and assert their expected behaviours. The usage of the Analyzer module can be seen for the thermal behaviours [here](https://github.com/ARM-software/bart/blob/master/notebooks/thermal/Thermal.ipynb)

# Scheduler Assertions

Enables assertion and the calculation of the following parameters:

### Runtime

The total time that the task spent on a CPU executing.

### Switch

Assert that a task switched between CPUs/Clusters in a given window of time

### Duty Cycle

The ratio of the execution time to the total time.

### Period

The average difference between two switch-in or two switch-out events of a task

### First CPU

The first CPU that a task ran on.

### Residency

Calculate and assert the total residency of a task on a CPU or cluster

### Examples

The Scheduler assertions also use TRAPpy's EventPlot to provide a kernelshark like timeline
for the tasks under consideration. (in IPython notebooks).

A notebook explaining the usage of the framework for asserting the deadline scheduler behaviours can be seen [here](https://rawgit.com/sinkap/0abbcc4918eb228b8887/raw/a1b4d6e0079f4ea0368d595d335bc340616501ff/SchedDeadline.html)


