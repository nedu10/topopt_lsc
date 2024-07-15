#include "topoptlib.h" // for wrapper
#include "Filter.h"
#include "LocalVolume.h"
#include "LinearElasticity.h"
#include "MMA.h"
#include "MPIIO.h"
#include "TopOpt.h"
#include "mpi.h"
#include <petsc.h>

#include <string>
#include <fstream>
#include <iostream>


/*
Modified: Thijs Smit, Dec 2020

Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013

Updated: June 2019, Niels Aage
Copyright (C) 2013-2019,

Disclaimer:
The authors reserves all rights but does not guaranty that the code is
free from errors. Furthermore, we shall not be liable in any event
caused by the use of the program.
*/

static char help[] = "3D TopOpt using KSP-MG on PETSc's DMDA (structured grids) \n";

PetscErrorCode DMDAGetElements_3D(DM dm, PetscInt* nel, PetscInt* nen, const PetscInt* e[]) {
    PetscErrorCode ierr;
    DM_DA*         da = (DM_DA*)dm->data;
    PetscInt       i, xs, xe, Xs, Xe;
    PetscInt       j, ys, ye, Ys, Ye;
    PetscInt       k, zs, ze, Zs, Ze;
    PetscInt       cnt = 0, cell[8], ns = 1, nn = 8;
    PetscInt       c;
    if (!da->e) {
        if (da->elementtype == DMDA_ELEMENT_Q1) {
            ns = 1;
            nn = 8;
        }
        ierr = DMDAGetCorners(dm, &xs, &ys, &zs, &xe, &ye, &ze);
        CHKERRQ(ierr);
        ierr = DMDAGetGhostCorners(dm, &Xs, &Ys, &Zs, &Xe, &Ye, &Ze);
        CHKERRQ(ierr);
        xe += xs;
        Xe += Xs;
        if (xs != Xs)
            xs -= 1;
        ye += ys;
        Ye += Ys;
        if (ys != Ys)
            ys -= 1;
        ze += zs;
        Ze += Zs;
        if (zs != Zs)
            zs -= 1;
        da->ne = ns * (xe - xs - 1) * (ye - ys - 1) * (ze - zs - 1);
        PetscMalloc((1 + nn * da->ne) * sizeof(PetscInt), &da->e);
        for (k = zs; k < ze - 1; k++) {
            for (j = ys; j < ye - 1; j++) {
                for (i = xs; i < xe - 1; i++) {
                    cell[0] = (i - Xs) + (j - Ys) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[1] = (i - Xs + 1) + (j - Ys) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[2] = (i - Xs + 1) + (j - Ys + 1) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[3] = (i - Xs) + (j - Ys + 1) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[4] = (i - Xs) + (j - Ys) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    cell[5] = (i - Xs + 1) + (j - Ys) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    cell[6] = (i - Xs + 1) + (j - Ys + 1) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    cell[7] = (i - Xs) + (j - Ys + 1) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    if (da->elementtype == DMDA_ELEMENT_Q1) {
                        for (c = 0; c < ns * nn; c++)
                            da->e[cnt++] = cell[c];
                    }
                }
            }
        }
    }
    *nel = da->ne;
    *nen = nn;
    *e   = da->e;
    return (0);
}

// Output the results to vtr files
PetscErrorCode outputPoints(const char *name, DM nd, Vec U, Vec xp) {

    PetscErrorCode ierr;

    PetscViewer viewer;

    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD, name, FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);

    ierr = DMView(nd, viewer);
    CHKERRQ(ierr);

    PetscObjectSetName((PetscObject)U,"U");
    ierr = VecView(U, viewer);
    CHKERRQ(ierr);

    PetscObjectSetName((PetscObject)xp,"xPhysPoints");
    ierr = VecView(xp, viewer);
    CHKERRQ(ierr);

    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);

    return ierr;
}

// solve function
int solve(DataObj data) {

    PetscMPIInt rank;

    // Zero run time input data...
    int argc = 0;
    char **argv = 0;

    // Error code for debugging
    PetscErrorCode ierr;

    // Initialize PETSc / MPI and pass input arguments to PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, help);

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt nx = 14;  // Number of nodes in x-direction
    PetscInt ny = 7;  // Number of nodes in y-direction
    PetscInt nz = 7;

    DM da;

    DMBoundaryType bx = DM_BOUNDARY_NONE;
    DMBoundaryType by = DM_BOUNDARY_NONE;
    DMBoundaryType bz = DM_BOUNDARY_NONE;


    // Create the nodal mesh
    ierr = DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, DMDA_STENCIL_BOX, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                        1, 5, 0, 0, 0, &da);
    CHKERRQ(ierr);

    DMSetFromOptions(da);
    // Set up the distributed mesh
    ierr = DMSetUp(da); DMView(da, PETSC_VIEWER_STDOUT_WORLD);
    // Access local data
    Vec localVec;
    ierr = DMCreateGlobalVector(da, &localVec); 
    
    CHKERRQ(ierr);

    VecSet(localVec, 1.0);


    PetscInt        nel, nen;
    const PetscInt* necon;

    ierr = DMDAGetElements_3D(da, &nel, &nen, &necon);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "nel = %d\n", nel);
    PetscPrintf(PETSC_COMM_WORLD, "nen = %d\n", nen);

    // PetscPrintf(PETSC_COMM_WORLD, "necon:\n");
    // for (PetscInt i = 0; i < nel; i++) {
    //     PetscPrintf(PETSC_COMM_WORLD, "Element %d: ", i);
    //     for (PetscInt j = 0; j < nen; j++) {
    //         PetscPrintf(PETSC_COMM_WORLD, "%d ", necon[i * nen + j]);
    //     }
    //     PetscPrintf(PETSC_COMM_WORLD, "\n");
    // }

    PetscInt dof;
    ierr = VecGetBlockSize(localVec, &dof);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "dof = %d\n", dof);

    PetscInt mx, my, mz;
    ierr = DMDAGetCorners(da, NULL, NULL, NULL, &mx, &my, &mz);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "mx = %d\n", mx);
    PetscPrintf(PETSC_COMM_WORLD, "my = %d\n", my);
    PetscPrintf(PETSC_COMM_WORLD, "mz = %d\n", mz);

    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    int step_size = static_cast<int>(floor(nx/4));
    PetscPrintf(PETSC_COMM_WORLD, "step_size = %d\n", step_size);

    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {

                PetscInt row = (i - info.gxs) + (j - info.gys) * (info.gxm) + (k - info.gzs) * (info.gxm) * (info.gym);
                // PetscPrintf(PETSC_COMM_WORLD, "row = %d\n", row);

                if(i < step_size){
                    PetscScalar value1 = 0.8;
                    VecSetValuesLocal(localVec, 1, &row, &value1, INSERT_VALUES);
                    // PetscPrintf(PETSC_COMM_WORLD, "i = %d\n", i);
                } else if(i < (step_size * 2)){
                     PetscScalar value2 = 0.6;
                     VecSetValuesLocal(localVec, 1, &row, &value2, INSERT_VALUES);
                    //  PetscPrintf(PETSC_COMM_WORLD, "i = %d\n", i);
                } else if(i < (step_size * 3)){
                     PetscScalar value3 = 0.4;
                     VecSetValuesLocal(localVec, 1, &row, &value3, INSERT_VALUES);
                    //  PetscPrintf(PETSC_COMM_WORLD, "i = %d\n", i);
                } else {
                     PetscScalar value4 = 0.2;
                     VecSetValuesLocal(localVec, 1, &row, &value4, INSERT_VALUES);
                    //  PetscPrintf(PETSC_COMM_WORLD, "i = %d\n", i);
                }
            }
        }
    }

    ierr = VecAssemblyBegin(localVec);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(localVec);
    CHKERRQ(ierr);



    PetscViewer viewer;
    ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    CHKERRQ(ierr);

    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);
    CHKERRQ(ierr);

    ierr = PetscViewerSetUp(viewer);
    CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer, "Local Vector:\n");
    CHKERRQ(ierr);

    ierr = VecView(localVec, viewer);
    CHKERRQ(ierr);

    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);


    


    PetscInt start, end;
    ierr = DMDAGetGhostCorners(da, &start, NULL, NULL, &end, NULL, NULL);
    CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "start = %d\n", start);
    PetscPrintf(PETSC_COMM_WORLD, "end = %d\n", end);



// Free resources
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = VecDestroy(&localVec); CHKERRQ(ierr);



    // PetscPrintf(PETSC_COMM_WORLD, "######################## Final output ########################\n");
    // //PetscPrintf(PETSC_COMM_WORLD, "# Porosity: %f\n", Poro);
    // PetscPrintf(PETSC_COMM_WORLD, "# Final compliance: %f\n", opt->fx / opt->fscale);
    // PetscPrintf(PETSC_COMM_WORLD, "# Total Volume: %f\n", xPhys_sum);
    // PetscPrintf(PETSC_COMM_WORLD, "# Mesh resolution: %f\n", opt->xc[1] / (opt->nxyz[0] - 1));
    // PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");

    
    // Finalize PETSc / MPI
    PetscFinalize();

    return 1;
}
