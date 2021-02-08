#!/bin/sh

platform_id=0   # Apple
device_id=1     # Intel(R) Iris(TM) Plus Graphics 640
package_id=0    # MNIST
config=1        # CNN
mode=1          # test
size=100        # not used in test

python ./main.py $platform_id $device_id $package_id $config $mode
