#!/bin/bash

# Author: Thijs Smit, Dec 2020
# Copyright (C) 2020 ETH Zurich

# Disclaimer:
# The authors reserves all rights but does not guaranty that the code is
# free from errors. Furthermore, we shall not be liable in any event
# caused by the use of the program.

### SETUP ENVIRONMENT ON ETH EULER
# by hand: env2lmod
# to use pre-installed petsc version: module load gcc/4.8.5 cmake/3.16.5 openmpi/3.0.1 petsc/3.10.5 python/3.7.4
module load gcc/4.8.5 cmake/3.16.5 openmpi/3.0.1 python/3.7.4

# to use petsc in PETSC_DIR
# export PETSC_ARCH=arch-linux2-c-opt # change path
# export PETSC_DIR=$home/petsc # change path

# export PETSC_DIR=/home/cifedior/Desktop/MASTERS/THESIS/CODE/petsc-3.14.1
export PETSC_DIR=/home/phediora/projects/def-osezua/phediora/large-scale-computing/petsc-3.14.1
export PETSC_ARCH=arch-linux-c-opt

# Print loaded modules
module list

# remove old files
rm -rf topoptlib.so
rm -rf build
mkdir build
cd build
cmake ..
make
cd ..

# RUN
cd $SCRATCH;

id=`date '+%Y%m%d_%H:%M:%S_test'`;
#echo $id

mkdir $id
cd $id

cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib.so .
cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib/test/test_beam.py .
cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib/test/test_multiload.py .
cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib/test/test_continuation.py .
cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib/test/test_projection.py .
cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib/test/test_bracket.py .
cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib/test/test_sphere.py .
cp ../../../home/thsmit/TopOpt_in_PETSc_wrapped_in_Python/topoptlib/test/test_michell.py .


# ADJUSTABLE PARAMETERS
EULER_MEMORY="2000"
NCPU=8
WALL_TIME="1:00"

# FUNCTION CALL
bsub -n ${NCPU} -W ${WALL_TIME} -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python test_beam.py
bsub -n ${NCPU} -W ${WALL_TIME} -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python test_multiload.py
bsub -n ${NCPU} -W ${WALL_TIME} -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python test_continuation.py
bsub -n ${NCPU} -W ${WALL_TIME} -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python test_projection.py
bsub -n ${NCPU} -W ${WALL_TIME} -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python test_bracket.py
bsub -n ${NCPU} -W ${WALL_TIME} -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python test_sphere.py

# ADJUSTABLE PARAMETERS
EULER_MEMORY="4000"
NCPU=32
WALL_TIME="1:00"

bsub -n ${NCPU} -W ${WALL_TIME} -R "rusage[mem=${EULER_MEMORY}]" mpirun -n ${NCPU} python test_michell.py

# VIEW JOBS
bjobs
