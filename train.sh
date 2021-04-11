#!/bin/sh

platform_id=1   # Thread Ripper
device_id=0     #
package_id=0    # MNIST
config=0        # FC
mode=0          # train
size=5000       # size of batch

python ./main.py $platform_id $device_id $package_id $config $mode $size
