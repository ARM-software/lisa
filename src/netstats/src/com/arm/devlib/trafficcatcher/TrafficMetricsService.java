package com.arm.devlib.netstats;

import java.lang.InterruptedException;
import java.lang.System;
import java.lang.Thread;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import android.app.Activity;
import android.app.IntentService;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.net.TrafficStats;
import android.os.Bundle;
import android.util.Log;

class TrafficPoller implements Runnable {

    private String tag;
    private int period;
    private PackageManager pm;
    private static String TAG = "TrafficMetrics";
    private List<String> packageNames;
    private Map<String, Map<String, Long>> previousValues;

    public TrafficPoller(String tag, PackageManager pm, int period, List<String> packages) {
        this.tag = tag;
        this.pm = pm;
        this.period = period;
        this.packageNames = packages;
        this.previousValues = new HashMap<String, Map<String, Long>>();
    }

    public void run() {
        try {
            while (true) {
                Thread.sleep(this.period);
                getPakagesInfo();
                if (Thread.interrupted()) {
                    throw new InterruptedException();
                }
            }
        } catch (InterruptedException e) {
        }
    }

    public void getPakagesInfo() {
        List<ApplicationInfo> apps;
        if (this.packageNames == null) {
            apps = pm.getInstalledApplications(0);
            for (ApplicationInfo app : apps) {
            }
        } else {
            apps = new ArrayList<ApplicationInfo>();
            for (String packageName : packageNames) {
                try {
                    ApplicationInfo info = pm.getApplicationInfo(packageName,  0);
                    apps.add(info);
                } catch (PackageManager.NameNotFoundException e) {
                    e.printStackTrace();
                }
            }
        }

        for (ApplicationInfo appInfo : apps) {
            int uid = appInfo.uid;
            String name = appInfo.packageName;
            long time =  System.currentTimeMillis();
            long received = TrafficStats.getUidRxBytes(uid);
            long sent = TrafficStats.getUidTxBytes(uid);

            if (!this.previousValues.containsKey(name)) {
                this.previousValues.put(name, new HashMap<String, Long>());
                this.previousValues.get(name).put("sent", sent);
                this.previousValues.get(name).put("received", received);
                Log.i(this.tag, String.format("INITIAL \"%s\" TX: %d RX: %d", 
                                              name, sent, received));
            } else {
                long previosSent = this.previousValues.get(name).put("sent", sent);
                long previosReceived = this.previousValues.get(name).put("received", received);
                Log.i(this.tag, String.format("%d \"%s\" TX: %d RX: %d", 
                                              time, name, 
                                              sent - previosSent, 
                                              received - previosReceived));
            }
        }
    }
}

public class TrafficMetricsService extends IntentService {

    private static String TAG = "TrafficMetrics";
    private Thread thread;
    private static int defaultPollingPeriod = 5000;

    public TrafficMetricsService() {
            super("TrafficMetrics");
    }

    @Override
    public void onHandleIntent(Intent intent) {
        List<String> packages = null;
        String runTag = intent.getStringExtra("tag");
        if (runTag == null) {
            runTag = TAG;
        }
        String packagesString = intent.getStringExtra("packages");
        int pollingPeriod = intent.getIntExtra("period", this.defaultPollingPeriod);
        if (packagesString != null) {
            packages = new ArrayList<String>(Arrays.asList(packagesString.split(",")));
        } 

        if (this.thread != null) {
            Log.e(runTag, "Attemping to start when monitoring is already in progress.");
            return;
        }
        this.thread = new Thread(new TrafficPoller(runTag, getPackageManager(), pollingPeriod, packages));
        this.thread.start();
    }
}
