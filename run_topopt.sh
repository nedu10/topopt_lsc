#!/bin/bash

# Author: Thijs Smit, Dec 2020
# Copyright (C) 2020 ETH Zurich

# Disclaimer:
# The authors reserves all rights but does not guaranty that the code is
# free from errors. Furthermore, we shall not be liable in any event
# caused by the use of the program.

## SETUP ENVIRONMENT on ETH EULER
# by hand: env2lmod

## For using the standard PETSc installation on ETH EULER:
#module load gcc/4.8.5 cmake/3.16.5 openmpi/3.0.1 petsc/3.10.5 python/3.7.4

## For using your own PETSc installation in $HOME:
module load gcc/4.8.5 cmake/3.16.5 openmpi/3.0.1 python/3.7.4
# export PETSC_ARCH=arch-linux2-c-opt
# export PETSC_DIR=$home/petsc
# export PETSC_DIR=/home/cifedior/Desktop/MASTERS/THESIS/CODE/petsc-3.14.1
export PETSC_DIR=/home/phediora/projects/def-osezua/phediora/large-scale-computing/petsc-3.14.1
export PETSC_ARCH=arch-linux-c-opt

## Print loaded modules
module list

## User input, do you want to compile or not
echo Press 1 for compiling, otherwise press 0...
read var

## Compile or not
if [ $var -eq 1 ]
    then
        rm -rf topoptlib.so
        rm -rf build
        mkdir build
        cd build
        cmake ..
        make
        cd ..
fi


###### RUNNING

# Go to the working directory
cd $SCRATCH
pwd

# Name of the new folder you store your files in
id=`date '+%Y%m%d_%H:%M:%S'`;
echo 'Name of new folder:'
echo $id

mkdir $id
cd $id

# Print current working directory
echo 'Current working directory: '
pwd

## Copy files needed to run simulation to current working directory
cp $HOME/TopOpt_in_PETSc_wrapped_in_Python/topoptlib.so .
cp $HOME/TopOpt_in_PETSc_wrapped_in_Python/examples/$1 .

## ETH EULER settings
EULER_MEMORY="4000"
NCPU=128
WALL_TIME="12:00"

# FUNCTION CALL
bsub -n ${NCPU} -W ${WALL_TIME} -R ib -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python $1

# VIEW JOBS
bjobs
