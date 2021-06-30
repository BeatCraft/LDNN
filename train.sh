#!/bin/sh

platform_id=0   # Apple
device_id=1     # Intel(R) Iris(TM) Plus Graphics 640
package_id=1    # CIFAR-10
config=1        # FC:0, CNN:1
mode=0          # train
size=500

python ./main.py $platform_id $device_id $package_id $config $mode $size

