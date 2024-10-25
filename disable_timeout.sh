#!/bin/bash

echo N > /sys/kernel/debug/gpu.0/timeouts_enabled
timer=`cat /sys/kernel/debug/gpu.0/timeouts_enabled`
echo "Timeout disabled" 
