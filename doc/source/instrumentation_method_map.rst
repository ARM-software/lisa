Instrumentation Signal-Method Mapping
=====================================

.. _instrumentation_method_map:

Instrument methods get automatically hooked up to signals based on their names. Mostly, the method
name correponds to the name of the signal, however there are a few convienience aliases defined
(listed first) to make  easier to relate instrumenation code to the workload execution model.

======================================== =========================================
method name                              signal                                   
======================================== =========================================
initialize                               run-init-signal                          
setup                                    successful-workload-setup-signal         
start                                    before-workload-execution-signal         
stop                                     after-workload-execution-signal          
process_workload_result                  successful-iteration-result-update-signal
update_result                            after-iteration-result-update-signal     
teardown                                 after-workload-teardown-signal           
finalize                                 run-fin-signal                           
on_run_start                             start-signal                             
on_run_end                               end-signal                               
on_workload_spec_start                   workload-spec-start-signal               
on_workload_spec_end                     workload-spec-end-signal                 
on_iteration_start                       iteration-start-signal                   
on_iteration_end                         iteration-end-signal                     
before_initial_boot                      before-initial-boot-signal               
on_successful_initial_boot               successful-initial-boot-signal           
after_initial_boot                       after-initial-boot-signal                
before_first_iteration_boot              before-first-iteration-boot-signal       
on_successful_first_iteration_boot       successful-first-iteration-boot-signal   
after_first_iteration_boot               after-first-iteration-boot-signal        
before_boot                              before-boot-signal                       
on_successful_boot                       successful-boot-signal                   
after_boot                               after-boot-signal                        
on_spec_init                             spec-init-signal                         
on_run_init                              run-init-signal                          
on_iteration_init                        iteration-init-signal                    
before_workload_setup                    before-workload-setup-signal             
on_successful_workload_setup             successful-workload-setup-signal         
after_workload_setup                     after-workload-setup-signal              
before_workload_execution                before-workload-execution-signal         
on_successful_workload_execution         successful-workload-execution-signal     
after_workload_execution                 after-workload-execution-signal          
before_workload_result_update            before-iteration-result-update-signal    
on_successful_workload_result_update     successful-iteration-result-update-signal
after_workload_result_update             after-iteration-result-update-signal     
before_workload_teardown                 before-workload-teardown-signal          
on_successful_workload_teardown          successful-workload-teardown-signal      
after_workload_teardown                  after-workload-teardown-signal           
before_overall_results_processing        before-overall-results-process-signal    
on_successful_overall_results_processing successful-overall-results-process-signal
after_overall_results_processing         after-overall-results-process-signal     
on_error                                 error_logged                             
on_warning                               warning_logged                           
======================================== =========================================


The names above may be prefixed with one of pre-defined prefixes to set the priority of the
Instrument method realive to other callbacks registered for the signal (within the same priority
level, callbacks are invoked in the order they were registered). The table below shows the mapping
of the prifix to the corresponding priority:

=========== ========
prefix      priority
=========== ========
very_fast\_       20
fast\_            10
normal\_           0
slow\_           -10
very_slow\_      -20
=========== ========

