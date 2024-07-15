#!/usr/bin/env python3

# Author: Thijs Smit, Dec 2020
# Copyright (C) 2020 ETH Zurich

# Disclaimer:
# The authors reserves all rights but does not guaranty that the code is
# free from errors. Furthermore, we shall not be liable in any event
# caused by the use of the program.

import numpy as np

import topoptlib

data = topoptlib.Data()
data.structuredGrid((0.0, 2.0, 0.0, 1.0, 0.0, 1.0), (65, 33, 33))
Emin, Emax, nu, Dens, penal = 1.0e-9, 1.0, 0.3, 1.0, 3.0
data.material(Emin, Emax, nu, Dens, penal)
data.projection(64.0, 1.0, 0.05)
data.filter(1, 0.08)
data.mma(40, 0.01)
data.loadcases(1)
data.bc(0, 1, [0], [0.0], [0, 1, 2], [0.0, 0.0, 0.0], 0)
data.bc(0, 2, [0, 2], [2.0, 0.0], [2], [-0.001], 0)
data.bc(0, 2, [0, 1, 2], [2.0, 0.0, 0.0], [2], [-0.0005], 0)
data.bc(0, 2, [0, 1, 2], [2.0, 1.0, 0.0], [2], [-0.0005], 0)

materialvolumefraction = 0.12
nEl = data.nElements


def objective(comp, sumXp, volfrac):
    return comp


def sensitivity(xp, uKu, penal):
    return -1.0 * penal * np.power(xp, (penal - 1)) * (Emax - Emin) * uKu


def constraint(comp, sumXp, volfrac):
    return sumXp / nEl - materialvolumefraction


def constraintSensitivity(xp, uKu, penal):
    return 1.0 / nEl


data.obj(objective)
data.objsens(sensitivity)
data.cons(constraint)
data.conssens(constraintSensitivity)
data.initialcondition(materialvolumefraction)


# get a function to run for testing
def Test(trueFX, runtime, memory):

    trueFX = int(trueFX * 1000) / 1000
    print("trueFX: ", trueFX)

    if trueFX == 0.489:
        open("test_projection SUCCESFULL", "w+")
    else:
        open("test_projection not succesfull!", "w+")


data.check(Test)

complete = data.solve()
