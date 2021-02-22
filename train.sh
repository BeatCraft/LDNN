#!/bin/sh

platform_id=0   # Apple
device_id=1     # Intel(R) Iris(TM) Plus Graphics 640
package_id=1    # cifar10
config=1        # cnn
mode=0          # train_test
size=200        # size of batch

python ./main.py $platform_id $device_id $package_id $config $mode $size

