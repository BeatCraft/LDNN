#!/bin/sh

platform_id=1 # Thraed Ripper
device_id=0 #
package_id=0 # MNIST
#package_id=1 #CIFAR-10
config=0 # FC
#config=1 # CNN
mode=1 # test

python ./main.py $platform_id $device_id $package_id $config $mode
