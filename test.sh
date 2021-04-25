#!/bin/sh

platform_id=1   # Thraed Ripper
device_id=0     #
package_id=1    # MNIST
config=0        # FC
mode=1          # test
size=5000       # size of batch

python ./main.py $platform_id $device_id $package_id $config $mode $size


