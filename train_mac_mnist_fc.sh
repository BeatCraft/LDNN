#!/bin/sh

platform_id=0   # Apple
device_id=1     # Intel(R) Iris(TM) Plus Graphics 640
package_id=0    # MNIST
config=0        # FC
mode=0          # train
size=100        # size of batch

python ./main.py $platform_id $device_id $package_id $config $mode $size

