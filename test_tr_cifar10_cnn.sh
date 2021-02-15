#!/bin/sh

platform_id=1   # Thraed Ripper
device_id=0     #
package_id=1    # cifar-10
config=1        # cnn
mode=1          # test
size=100        # not used in test

python ./main.py $platform_id $device_id $package_id $config $mode $size


