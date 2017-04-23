#!/usr/bin/env python

from collections import namedtuple
import trappy

EventRow = namedtuple('EventRow', ['time', 'event', 'row'])
LatRow = namedtuple('LatRow', ['pid', 'switch_row', 'wake_row', 'last_wake_row', 'running', 'wake_pend', 'latency'])

def printsw(time, data):
    print str(time) + ": " + str(data)

def print_event_rows(rows, start, end):
	print "time".ljust(20) + "\t" + "event".ljust(10) + "\tpid" + "\tprevpid" + "\tnextpid"
	for r in rows:
		if r.time < start or r.time > end:
			continue
		if r.event == "switch":
			nextpid =  r.row['next_pid']
			prevpid = r.row['prev_pid']
			pid = -1
		elif r.event == "wake":
			pid = r.row['pid']
			nextpid = -1
			prevpid = -1
		else:
			raise RuntimeError("unknown event seen")
		print str(r.time).ljust(20)[:20] + "\t" + r.event.ljust(10) + "\t" + str(pid) + "\t" + str(prevpid) + "\t" + str(nextpid)

path_to_html = "/home/joelaf/repo/catapult/systrace/trace_sf-dispsync_rt_pull_fail_1504ms.html"

# Hash table of pid -> LatRow named tuple
latpids = {}
dpid = 21047
debugg = False

def switch_cb(time, data):
    row = data
    event = "switch"

    prevpid = row['prev_pid']
    nextpid = row['next_pid']

    r = EventRow(time=time, event="switch", row=data)

    debug = debugg and (prevpid == dpid or nextpid == dpid)

    if (debug):
        print "\nProcessing switch nextpid=" + str(nextpid) + " prevpid=" + str(prevpid) + \
              " time=" + str(time)

    # prev pid processing (switch out)
    if latpids.has_key(prevpid):
        if latpids[prevpid].running == 1:
            latpids[prevpid] = latpids[prevpid]._replace(running=0)
        if latpids[prevpid].wake_pend == 1:
            print "Impossible: switch out during wake_pend " + str(r)
            raise RuntimeError("error")

    if not latpids.has_key(nextpid):
        return

    # nextpid processing  (switch in)
    pid = nextpid
    if latpids[pid].running == 1:
        print "INFO: previous pid switch-out not seen for an event, ignoring\n" + str(r)
        return
    latpids[pid] = latpids[pid]._replace(running=1)

    # Ignore latency calc for next-switch events for which wake never seen
    # They are still valid in this scenario because of preemption
    if latpids[pid].wake_pend == 0:
        if debug:
            print "wake_pend = 0, doing nothing"
        return

    if debug:
        print "recording"
    # Measure latency
    lat = time - latpids[pid].last_wake_row.time
    if lat > latpids[pid].latency:
        latpids[pid] = LatRow(pid, switch_row = r,
                              wake_row = latpids[pid].last_wake_row,
                              last_wake_row=None, latency=lat, running=1, wake_pend=0)
        return
    latpids[pid] = latpids[pid]._replace(running=1, wake_pend=0)

def wake_cb(time, data):
    row = data
    event = "wake"
    pid = row["pid"]
    debug = debugg and (pid == dpid)

    r = EventRow(time=time, event="wake", row=data)

    if (debug):
        print "\nProcessing wake pid=" + str(pid) + " time=" + str(time)
    if  not latpids.has_key(pid):
        latpids[pid] = LatRow(pid, switch_row=None, wake_row=None,
                last_wake_row = r, running=0, latency=-1, wake_pend=1)
        if debug:
            print "new wakeup"
        return

    if latpids[pid].running == 1 or latpids[pid].wake_pend == 1:
        if debug:
            print "already running or wake_pend"
            # Task already running or a wakeup->switch pending, ignore
        return

    if debug:
        print "recording wake"
    latpids[pid] = latpids[pid]._replace(last_wake_row = r, wake_pend=1)

systrace_obj = trappy.SysTrace(name="systrace", path=path_to_html, \
        scope="sched", events=["sched_switch", "sched_wakeup", "sched_waking"],
        event_callbacks={ "sched_switch": switch_cb, "sched_wakeup": wake_cb })

"""
df = systrace_obj.sched_switch.data_frame
# df = df[(df.next_prio < 100) | (df.prev_prio < 100)]
switch_df = df

df = systrace_obj.sched_wakeup.data_frame
# df = df[df.prio < 100]
wake_df = df

# Build list of rows
switch_rows = [EventRow(time=i, event="switch", row=row) for i, row in switch_df.iterrows()]
wake_rows = [EventRow(time=i, event="wake", row=row) for i, row in wake_df.iterrows()]
rows = switch_rows + wake_rows
rows =  sorted(rows, key=lambda r: r.time)
"""

# Print PID, latency, start, end, sort
result = sorted(latpids.items(), key=lambda x: x[1].latency, reverse=True)

print "PID".ljust(10) + "\t" + "name".ljust(20) + "\t" + "latency (secs)".ljust(20) + \
      "\t" + "start time".ljust(20) + "\t" + "end time".ljust(20)
for r in result:
	l = r[1] # LatRow named tuple
	if l.pid != r[0]:
		raise RuntimeError("BUG: pid inconsitency found") # Sanity check
	print str(r[0]).ljust(10) + "\t" + str(l.wake_row.row['comm']).ljust(20) + "\t" + \
		  str(l.latency).ljust(20)[:20] + "\t" + str(l.wake_row.time).ljust(20)[:20] + \
		  "\t" + str(l.switch_row.time).ljust(20)[:20]

