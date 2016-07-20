## LISA - Linux Interactive System Analysis

### What is LISA?

The LISA is a toolkit that supports regression testing and interactive analysis of workload behaviours in Linux environment. LISA helps Linux kernel developers to measure the impact of modifications in core parts of the kernel. The focus is on the scheduler, power management and thermal frameworks. Nevertheless, LISA is generic and can be used for other purposes too.

LISA provides an API for modelling use-cases of interest and developing regression tests for use-cases. A ready made set of test-cases to support regression testing of core kernel features is provided. In addition, LISA uses the excellent IPython Notebook framework and a set of example notebooks for live experiments on a target platform.

### Motivations

Main goals of LISA are:

* Support study of existing behaviours (i.e. *"how does PELT work?"*)
* Support analysis of new code being developed (i.e. *"what is the impact on existing code?"*)
* Get insights on what's not working and possibly chase down why
* Share reproducible experiments by means of a **common language** that:
    * is **flexible enough** to reproduce the same experiment on different targets
    * **simplifies** generation and execution of well defined workloads
    * **defines** a set of metrics to evaluate kernel behaviours
    * **enables** kernel developers to easily post process data to produce statistics and plots


### Overall View
![LISA Overall View](https://cloud.githubusercontent.com/assets/63746/16997941/7644ea56-4eaf-11e6-81af-310c3f0e2ef6.png)

### External Links
* Linux Integrated System Analysis (LISA) & Friends
  [Slides](http://events.linuxfoundation.org/sites/events/files/slides/ELC16_LISA_20160326.pdf) and
  [Video](https://www.youtube.com/watch?v=yXZzzUEngiU)