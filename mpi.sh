#!/bin/sh
# $1
#mpiexec -host localhost:2 -n 2 python ./mpi_test.py

if [ $# == 0 ] ; then
    echo "no number pf process"
    echo "ie. mpi.sh 2"
elif [ $# == 1 ] ; then
    ls $1
    mpiexec -host localhost:$1 -n $1 python ./mpi_test.py
else
    echo "too many options"
fi

