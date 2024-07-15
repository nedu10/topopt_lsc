#!/usr/bin/env python3

# Author: Thijs Smit, Dec 2020
# Copyright (C) 2020 ETH Zurich

# Disclaimer:
# The authors reserves all rights but does not guaranty that the code is
# free from errors. Furthermore, we shall not be liable in any event
# caused by the use of the program.


def main():
    import numpy as np

    import topoptlib

    # step 1:
    # Create data class to store input data
    data = topoptlib.Data()

    # step 2:
    # define input data
    # mesh: (domain: x, y, z)(mesh: number of nodes)
    # data.structuredGrid(
    #    (0.0, 2.0, 0.0, 1.0, 0.0, 1.0), (65, 33, 33)
    # )

    data.structuredGrid((0.0, 2.0, 0.0, 1.0, 0.0, 1.0), (33, 17, 17))

    # data.structuredGrid(
    #   (0.0, 2.0, 0.0, 1.0, 0.0, 1.0), (257, 129, 129)
    # )

    # Optional printing:
    # print(data.nNodes)
    # print(data.nElements)
    # print(data.nDOF)

    # material: (Emin, Emax, nu, penal)
    Emin, Emax, nu, Dens, penal = 1.0e-6, 1.0, 0.3, 1.0, 3.0
    data.material(Emin, Emax, nu, Dens, penal)

    # setup heavyside projection filter (betaFinal, betaInit, eta)
    # data.projection(64.0, 1.0, 0.5)

    # filter: (type, radius)
    # filter types: sensitivity = 0, density = 1
    # using 0.08, 0.04 or 0.02
    data.filter(1, 0.08)

    # optimizer: (maxIter, tol)
    data.mma(400, 0.01)
    # data.mma(8000, 0.01)

    # loadcases: (# of loadcases)
    data.loadcases(1)

    # bc: (loadcase, type, [checker: dof index], [checker: values], [setter: dof index], [setter: values], parametrization)
    # bc: (loadcase, type, [coordinate axis], [coordinate value], [coordinate axis], [bc value], parametrization)
    data.bc(0, 1, [0], [0.0], [0, 1, 2], [0.0, 0.0, 0.0], 0)
    data.bc(0, 2, [0, 2], [2.0, 0.0], [2], [-0.001], 0)
    data.bc(0, 2, [0, 1, 2], [2.0, 0.0, 0.0], [2], [-0.0005], 0)
    data.bc(0, 2, [0, 1, 2], [2.0, 1.0, 0.0], [2], [-0.0005], 0)

    materialvolumefraction = 0.12
    # materialvolumefraction = 0.24 # (3.f)
    # complicancetarget = 1.5
    nEl = data.nElements

    # Calculate the objective function, senitivity, constraint and constraint sensitivity
    def objective(comp, sumXp, volfrac):
        return comp  # for minimizing compliance (3.b)
        # return sumXp # for minimizing volume (3.c)

    def sensitivity(xp, uKu, penal):
        return -1.0 * penal * np.power(xp, (penal - 1)) * (Emax - Emin) * uKu
        # for minimizing compliance (3.b)
        # return 1.0 # for minimizing volume (3.c)

    def constraint(comp, sumXp, volfrac):
        # return 0.0 # Omitting total volume constraint (3.d)
        return (
            sumXp / nEl - materialvolumefraction
        )  # Total volume constraint for minimizing compliance (3.b)
        # return comp / complicancetarget - 1.0 # for minimizing volume (3.c)

    def constraintSensitivity(xp, uKu, penal):
        # return 0.0 # Omitting total volume constraint (3.d)
        return 1.0 / nEl  # Total volume constraint for minimizing compliance (3.b)
        # return (-1.0 * penal * np.power(xp, (penal - 1)) * (Emax - Emin) * uKu) / complicancetarget # for minimizing volume (3.c)

    # Callback implementatio
    data.obj(objective)
    data.objsens(sensitivity)

    # Define constraint
    data.cons(constraint)
    data.conssens(constraintSensitivity)

    # Use local volume constraint additionally
    # Local volume constraint input: (Rlocvol, alpha)
    data.localVolume(0.16, 0.12)

    # Volume constraint is standard, input (volume fraction)
    data.initialcondition(materialvolumefraction)

    # number of cores used
    nc = 128

    # get a function to run for testing
    def Test(trueFX, runtime, memory):
        trueFX = int(trueFX * 1000) / 1000
        CPUtime = int(np.round(runtime * nc, decimals=0))  # in seconds
        memory = int((memory * 0.000001) * nc)  # in mega bytes
        print("trueFX: ", trueFX)
        print("CPU time: ", CPUtime, " sec")
        print("Memory use: (extrapolated) ", memory, " MB")

    data.check(Test)

    # To output vtr files
    # By default the initial condition and the first 10 iterations
    # will be written to a vtr file
    # Specify for which interval you need vtr files after the first 10
    data.vtr(20)

    # step 3:
    # solve topopt problem with input data and wait for "complete" signal
    data.solve()


if __name__ == "__main__":
    main()
