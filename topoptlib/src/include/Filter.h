#ifndef __FILTER__
#define __FILTER__

#include <petsc.h>
//#include <petsc-private/dmdaimpl.h>
#include "PDEFilter.h"
#include <iostream>
#include <math.h>
#include <petsc/private/dmdaimpl.h>

/* -----------------------------------------------------------------------------
Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013
 Updated: June 2019, Niels Aage
 Copyright (C) 2013-2019,

This Filter implementation is licensed under Version 2.1 of the GNU
Lesser General Public License.

This MMA implementation is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This Module is distributed in the hope that it will be useful,implementation
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
-------------------------------------------------------------------------- */

class Filter {

  public:
    // Constructor
    //Filter(DM da_nodes, Vec x, PetscInt filterT, PetscScalar Rin);
    Filter(DM da_nodes, Vec x, Vec xPassive, PetscInt filterT, PetscScalar Rin);

    // Destructor
    ~Filter();

    // Filter design variables
    PetscErrorCode FilterProject(Vec x, Vec xTilde, Vec xPhysEro, Vec xPhys, Vec xPhysDil, PetscBool projectionFilter, PetscScalar beta,
                                 PetscScalar eta);

    // Filter design variables for Robust approach
    PetscErrorCode FilterProjectRobust(Vec x, Vec xTilde, Vec xPhysEro, Vec xPhys, Vec xPhysDil, PetscBool projectionFilter, PetscScalar beta,
                                 PetscScalar eta, PetscScalar delta);


    // Filter the sensitivities
    PetscErrorCode Gradients(Vec x, Vec xTilde, Vec dfdx, PetscInt m, Vec* dgdx, PetscBool projectionFilter,
                             PetscScalar beta, PetscScalar eta);

    // Filter the sensitivities for Robust approach
    PetscErrorCode GradientsRobust(Vec x, Vec xTilde, Vec dfdx, PetscInt m, Vec* dgdx, PetscBool projectionFilter,
                             PetscScalar beta, PetscScalar eta, PetscScalar delta);

    // COntinuation for projection filter
    PetscBool IncreaseBeta(PetscReal* beta, PetscReal betaFinal, PetscScalar gx, PetscInt itr, PetscReal ch);

    // Measure of non-discreteness
    PetscScalar GetMND(Vec x);

    PetscErrorCode UpdatexPhys(Vec x, Vec xp);

    PetscErrorCode SetUpT(DM da_nodes);

  private:
    // Standard density/sensitivity filter matrix
    Mat H;  // Filter matrix
    Vec Hs; // Filter "sum weight" (normalization factor) vector
    Vec dx; // Projection filter chainrule correction
    Vec dx1; // Projection filter chainrule correction

    PetscInt    filterType;
    PetscScalar R;

    // Mesh used for standard filtering
    DM da_elem; // da for image-filter field mesh

    // PDE filtering
    PDEFilt* pdef; // PDE filter class

    // Setup datastructures for the filter
    PetscErrorCode SetUp(DM da_nodes, Vec x, Vec xPassive);

    // Projection
    PetscErrorCode HeavisideFilter(Vec x, Vec y, PetscReal beta, PetscReal eta);
    PetscErrorCode ChainruleHeavisideFilter(Vec y, Vec x, PetscReal beta, PetscReal eta);

    // The HS projection
    inline PetscReal SmoothProjection(PetscReal x, PetscReal beta, PetscReal eta) {
        PetscReal y = (tanh(beta * eta) + tanh(beta * (x - eta))) / (tanh(beta * eta) + tanh(beta * (1.0 - eta)));
        return y;
    };
    // Chainrule contribution
    inline PetscReal ChainruleSmoothProjection(PetscReal x, PetscReal beta, PetscReal eta) {
        PetscReal dx = beta * (1.0 - pow(tanh(beta * (x - eta)), 2.0)) / (tanh(beta * eta) + tanh(beta * (1.0 - eta)));
        return dx;
    };

    // Routine that doesn't change the element type upon repeated calls
    PetscErrorCode DMDAGetElements_3D(DM dm, PetscInt* nel, PetscInt* nen, const PetscInt* e[]);
};

#endif
