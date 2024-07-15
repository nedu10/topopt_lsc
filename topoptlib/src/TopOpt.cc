#include "TopOpt.h"
#include <cmath>

//#include <Python.h>
#include <stdio.h>
//#include <petscvec.h>
#include "petscis.h"
#include <petscviewer.h>

/*
 Modified by: Thijs Smit, Dec 2020

 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013
 Updated: June 2019, Niels Aage
 Copyright (C) 2013-2019,

 Disclaimer:
 The authors reserves all rights but does not guaranty that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/

TopOpt::TopOpt(DataObj data) {

    m = 1;
    if (data.m > 1) {
        m = data.m;
    }

    x        = NULL;
    xPhys    = NULL;
    dfdx     = NULL;
    dgdx     = NULL;
    gx       = NULL;
    da_nodes = NULL;
    da_elem  = NULL;

    xPassive = NULL;
    indicator = NULL;

    xMMA = NULL;
    dfdxMMA = NULL;
    dgdxMMA = NULL;
    xminMMA = NULL;
    xmaxMMA = NULL;
    xoldMMA = NULL;

    xo1 = NULL;
    xo2 = NULL;
    U   = NULL;
    L   = NULL;

    // Robust Formulation
    xPhysEro   = NULL;
    xPhysDil   = NULL;

    // xPhys to point data for output to .vtr
    xPhysPoints = NULL;

    //graded porous structure
    gradedPorousAlpha = NULL;

    SetUp(data);
}

TopOpt::~TopOpt() {

    // Delete vectors
    if (x != NULL) {
        VecDestroy(&x);
    }
    if (xTilde != NULL) {
        VecDestroy(&xTilde);
    }
    if (xPhys != NULL) {
        VecDestroy(&xPhys);
    }
    if (dfdx != NULL) {
        VecDestroy(&dfdx);
    }
    if (dgdx != NULL) {
        VecDestroyVecs(m, &dgdx);
    }
    if (xold != NULL) {
        VecDestroy(&xold);
    }
    if (xmin != NULL) {
        VecDestroy(&xmin);
    }
    if (xmax != NULL) {
        VecDestroy(&xmax);
    }
    if (xPassive != NULL) {
        VecDestroy(&xPassive);
    }
    if (indicator != NULL) {
        VecDestroy(&indicator);
    }
    if (xMMA != NULL) {
        VecDestroy(&xMMA);
    }
    if (dfdxMMA != NULL) {
        VecDestroy(&dfdxMMA);
    }
    if (dgdxMMA != NULL) {
        VecDestroyVecs(m, &dgdxMMA);
    }
    if (xminMMA != NULL) {
        VecDestroy(&xminMMA);
    }
    if (xmaxMMA != NULL) {
        VecDestroy(&xmaxMMA);
    }
    if (xoldMMA != NULL) {
        VecDestroy(&xoldMMA);
    }
    if (da_nodes != NULL) {
        DMDestroy(&(da_nodes));
    }
    if (da_elem != NULL) {
        DMDestroy(&(da_elem));
    }

    // Delete constraints
    if (gx != NULL) {
        delete[] gx;
    }

    // mma restart method
    if (xo1 != NULL) {
        VecDestroy(&xo1);
    }
    if (xo2 != NULL) {
        VecDestroy(&xo2);
    }
    if (L != NULL) {
        VecDestroy(&L);
    }
    if (U != NULL) {
        VecDestroy(&U);
    }

	if (xPhysEro!=NULL) {
        VecDestroy(&xPhysEro);
    }
	if (xPhysDil!=NULL) {
        VecDestroy(&xPhysDil);
    }

    if (xPhysPoints!=NULL) {
        VecDestroy(&xPhysPoints);
    }

    // destroy vector for graded porous structure
    if (gradedPorousAlpha != NULL) {
        VecDestroy(&gradedPorousAlpha);
    }

}

// NO METHODS !
// PetscErrorCode TopOpt::SetUp(Vec CRAPPY_VEC){
PetscErrorCode TopOpt::SetUp(DataObj data) {
    PetscErrorCode ierr;

    //using namespace data;

    // SET DEFAULTS for FE mesh and levels for MG solver
    nxyz[0] = data.nxyz[0];
    nxyz[1] = data.nxyz[1];
    nxyz[2] = data.nxyz[2];
    //nxyz[0] = data.nxyz_w[0];
    //nxyz[1] = data.nxyz_w[1];
    //nxyz[2] = data.nxyz_w[2];
    xc[0]   = data.xc[0];
    xc[1]   = data.xc[1];
    xc[2]   = data.xc[2];
    xc[3]   = data.xc[3];
    xc[4]   = data.xc[4];
    xc[5]   = data.xc[5];
    //xc[6]   = data.xc[6];
    //xc[7]   = data.xc[7];
    //xc[8]   = data.xc[8];
    //xc[9]   = data.xc[9];
    //xc[10]   = data.xc[10];
    //xc[11]   = data.xc[11];
    //xc[0]   = data.xc_w[0];
    //xc[1]   = data.xc_w[1];
    //xc[2]   = data.xc_w[2];
    //xc[3]   = data.xc_w[3];
    //xc[4]   = data.xc_w[4];
    //xc[5]   = data.xc_w[5];
    //xc[6]   = data.xc_w[6];
    //xc[7]   = data.xc_w[7];
    //xc[8]   = data.xc_w[8];
    //xc[9]   = data.xc_w[9];
    //xc[10]   = data.xc_w[10];
    nu      = data.nu;
    nlvls   = 4;

    // SET DEFAULTS for optimization problems
    volfrac = data.volumefrac;
    volfracREF = data.volumefrac;
    maxItr  = data.maxIter;
    tol  = data.tol;
    rmin    = data.rmin;

    localVolumeStatus = PETSC_FALSE;
    if (data.localVolume_w == 1) {
        localVolumeStatus = PETSC_TRUE;
    }

    testStatus = PETSC_FALSE;
    if (data.test_w == 1) {
        testStatus = PETSC_TRUE;
    }

    // continuation of penalization
    // Status
    penalIni = 1.0;
    penalFin = 3.0;
    penal = data.penal;
    penalStep = 0.25;
    IterProj = 10;
    continuationStatus = PETSC_FALSE;
    if (data.continuation_w == 1) {
        continuationStatus = PETSC_TRUE;
        penalIni = data.penalinitial_w;
        penalFin = data.penalfinal_w;
        penalStep = data.stepsize_w;
        IterProj = data.iterProg_w;
    }

    Emin    = data.Emin;
    Emax    = data.Emax;
    filter  = data.filter; // 0=sens,1=dens,2=PDE
    Xmin    = 0.0;
    Xmax    = 1.0;
    movlim  = 0.2;
    //restart = PETSC_TRUE;
    restart = PETSC_FALSE;

    // xPassive Status
    xPassiveStatus = PETSC_FALSE;
    if (data.xPassive_w.size() > 1) {
        xPassiveStatus = PETSC_TRUE;
    }

    // Projection filter, treshold designvar
    projectionFilter = PETSC_FALSE;
    beta             = 1.0;
    betaFinal        = 64.0;
    eta              = 0.5;
    if (data.projection_w == 1) {
        projectionFilter = PETSC_TRUE;
        beta             = data.betaInit_w;
        betaFinal        = data.betaFinal_w;
        eta              = data.eta_w;
    }

     // Projection filter, treshold designvar
    robustStatus = PETSC_FALSE;
    if (data.robust_w == 1) {
        robustStatus = PETSC_TRUE;
        projectionFilter = PETSC_TRUE;
        beta             = data.betaInit_w;
        betaFinal        = data.betaFinal_w;
        eta              = data.eta_w;
        delta            = data.delta_w;
    }

    ierr = SetUpMESH(data);
    CHKERRQ(ierr);

    ierr = SetUpOPT(data);
    CHKERRQ(ierr);

    return (ierr);
}

PetscErrorCode TopOpt::SetUpMESH(DataObj data) {

    PetscErrorCode ierr;

    //using data::nxyz;

    // Read input from arguments
    PetscBool flg;

    // Physics parameters
    PetscOptionsGetInt(NULL, NULL, "-nx", &(nxyz[0]), &flg);
    PetscOptionsGetInt(NULL, NULL, "-ny", &(nxyz[1]), &flg);
    PetscOptionsGetInt(NULL, NULL, "-nz", &(nxyz[2]), &flg);
    PetscOptionsGetReal(NULL, NULL, "-xcmin", &(xc[0]), &flg);
    PetscOptionsGetReal(NULL, NULL, "-xcmax", &(xc[1]), &flg);
    PetscOptionsGetReal(NULL, NULL, "-ycmin", &(xc[2]), &flg);
    PetscOptionsGetReal(NULL, NULL, "-ycmax", &(xc[3]), &flg);
    PetscOptionsGetReal(NULL, NULL, "-zcmin", &(xc[4]), &flg);
    PetscOptionsGetReal(NULL, NULL, "-zcmax", &(xc[5]), &flg);
    PetscOptionsGetReal(NULL, NULL, "-penal", &penal, &flg);
    PetscOptionsGetInt(NULL, NULL, "-nlvls", &nlvls,
                       &flg); // NEEDS THIS TO CHECK IF MESH IS OK BEFORE PROCEEDING !!!!

    // Write parameters for the physics _ OWNED BY TOPOPT
    PetscPrintf(PETSC_COMM_WORLD, "##############################################"
                                  "##########################\n");
    PetscPrintf(PETSC_COMM_WORLD, "############################ FEM settings "
                                  "##############################\n");
    PetscPrintf(PETSC_COMM_WORLD, "# Number of nodes: (-nx,-ny,-nz):        (%i,%i,%i) \n", nxyz[0], nxyz[1], nxyz[2]);
    PetscPrintf(PETSC_COMM_WORLD, "# Number of degree of freedom:           %i \n", 3 * nxyz[0] * nxyz[1] * nxyz[2]);
    PetscPrintf(PETSC_COMM_WORLD, "# Number of elements:                    (%i,%i,%i) \n", nxyz[0] - 1, nxyz[1] - 1,
                nxyz[2] - 1);
    PetscPrintf(PETSC_COMM_WORLD, "# Dimensions: (-xcmin,-xcmax,..,-zcmax): (%f,%f,%f)\n", xc[1] - xc[0], xc[3] - xc[2],
                xc[5] - xc[4]);
    PetscPrintf(PETSC_COMM_WORLD, "# -nlvls: %i\n", nlvls);
    PetscPrintf(PETSC_COMM_WORLD, "##############################################"
                                  "##########################\n");

    // Check if the mesh supports the chosen number of MG levels
    PetscScalar divisor = PetscPowScalar(2.0, (PetscScalar)nlvls - 1.0);
    // x - dir
    if (std::floor((PetscScalar)(nxyz[0] - 1) / divisor) != (nxyz[0] - 1.0) / ((PetscInt)divisor)) {
        PetscPrintf(PETSC_COMM_WORLD, "MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD, "X - number of nodes %i is cannot be halfened %i times\n", nxyz[0], nlvls - 1);
        exit(0);
    }
    // y - dir
    if (std::floor((PetscScalar)(nxyz[1] - 1) / divisor) != (nxyz[1] - 1.0) / ((PetscInt)divisor)) {
        PetscPrintf(PETSC_COMM_WORLD, "MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD, "Y - number of nodes %i is cannot be halfened %i times\n", nxyz[1], nlvls - 1);
        exit(0);
    }
    // z - dir
    if (std::floor((PetscScalar)(nxyz[2] - 1) / divisor) != (nxyz[2] - 1.0) / ((PetscInt)divisor)) {
        PetscPrintf(PETSC_COMM_WORLD, "MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD, "Z - number of nodes %i is cannot be halfened %i times\n", nxyz[2], nlvls - 1);
        exit(0);
    }

    // Start setting up the FE problem
    // Boundary types: DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED,
    // DMDA_BOUNDARY_PERIODIC
    DMBoundaryType bx = DM_BOUNDARY_NONE;
    DMBoundaryType by = DM_BOUNDARY_NONE;
    DMBoundaryType bz = DM_BOUNDARY_NONE;

    // Stencil type - box since this is closest to FEM (i.e. STAR is FV/FD)
    DMDAStencilType stype = DMDA_STENCIL_BOX;

    // Discretization: nodes:
    // For standard FE - number must be odd
    // For periodic: Number must be even
    PetscInt nx = nxyz[0];
    PetscInt ny = nxyz[1];
    PetscInt nz = nxyz[2];

    // number of nodal dofs: Nodal design variable - NOT REALLY NEEDED
    PetscInt numnodaldof = 1;

    // Stencil width: each node connects to a box around it - linear elements
    PetscInt stencilwidth = 1;

    // Coordinates and element sizes: note that dx,dy,dz are half the element size
    PetscReal xmin = xc[0], xmax = xc[1], ymin = xc[2], ymax = xc[3], zmin = xc[4], zmax = xc[5];
    dx = (xc[1] - xc[0]) / (PetscScalar(nxyz[0] - 1));
    dy = (xc[3] - xc[2]) / (PetscScalar(nxyz[1] - 1));
    dz = (xc[5] - xc[4]) / (PetscScalar(nxyz[2] - 1));

    // Create the nodal mesh
    ierr = DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, stype, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                        numnodaldof, stencilwidth, 0, 0, 0, &(da_nodes));
    CHKERRQ(ierr);

    // Initialize
    DMSetFromOptions(da_nodes);
    DMSetUp(da_nodes);

    // Set the coordinates
    ierr = DMDASetUniformCoordinates(da_nodes, xmin, xmax, ymin, ymax, zmin, zmax);
    CHKERRQ(ierr);

    // Set the element type to Q1: Otherwise calls to GetElements will change to
    // P1 ! STILL DOESN*T WORK !!!!
    ierr = DMDASetElementType(da_nodes, DMDA_ELEMENT_Q1);
    CHKERRQ(ierr);

    // Create the element mesh: NOTE THIS DOES NOT INCLUDE THE FILTER !!!
    // find the geometric partitioning of the nodal mesh, so the element mesh will
    // coincide with the nodal mesh
    PetscInt md, nd, pd;
    ierr = DMDAGetInfo(da_nodes, NULL, NULL, NULL, NULL, &md, &nd, &pd, NULL, NULL, NULL, NULL, NULL, NULL);
    CHKERRQ(ierr);

    // vectors with partitioning information
    PetscInt* Lx = new PetscInt[md];
    PetscInt* Ly = new PetscInt[nd];
    PetscInt* Lz = new PetscInt[pd];

    // get number of nodes for each partition
    const PetscInt *LxCorrect, *LyCorrect, *LzCorrect;
    ierr = DMDAGetOwnershipRanges(da_nodes, &LxCorrect, &LyCorrect, &LzCorrect);
    CHKERRQ(ierr);

    // subtract one from the lower left corner.
    for (int i = 0; i < md; i++) {
        Lx[i] = LxCorrect[i];
        if (i == 0) {
            Lx[i] = Lx[i] - 1;
        }
    }
    for (int i = 0; i < nd; i++) {
        Ly[i] = LyCorrect[i];
        if (i == 0) {
            Ly[i] = Ly[i] - 1;
        }
    }
    for (int i = 0; i < pd; i++) {
        Lz[i] = LzCorrect[i];
        if (i == 0) {
            Lz[i] = Lz[i] - 1;
        }
    }

    // Create xPhysPoints vector
    DMCreateGlobalVector(da_nodes, &(xPhysPoints));

    // Create the element grid: NOTE CONNECTIVITY IS 0
    PetscInt conn = 0;
    ierr = DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, stype, nx - 1, ny - 1, nz - 1, md, nd, pd, 1, conn, Lx, Ly, Lz,
                        &(da_elem));
    CHKERRQ(ierr);

    // Initialize
    DMSetFromOptions(da_elem);
    DMSetUp(da_elem);

    // Set element center coordinates
    ierr = DMDASetUniformCoordinates(da_elem, xmin + dx / 2.0, xmax - dx / 2.0, ymin + dy / 2.0, ymax - dy / 2.0,
                                     zmin + dz / 2.0, zmax - dz / 2.0);
    CHKERRQ(ierr);

    // Clean up
    delete[] Lx;
    delete[] Ly;
    delete[] Lz;

    return (ierr);
}

PetscErrorCode TopOpt::SetUpOPT(DataObj data) {

    PetscErrorCode ierr;

    // ierr = VecDuplicate(CRAPPY_VEC,&xPhys); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da_elem, &xPhys);
    CHKERRQ(ierr);

    //creating alpha for the graded porous structure

    ierr = DMCreateGlobalVector(da_elem, &gradedPorousAlpha);
    CHKERRQ(ierr);

    VecSet(gradedPorousAlpha, 1.0);

    DMDALocalInfo info;
    DMDAGetLocalInfo(da_elem, &info);

    int step_size = static_cast<int>(floor((nxyz[0]-1)/10));

    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {

                PetscInt row = (i - info.gxs) + (j - info.gys) * (info.gxm) + (k - info.gzs) * (info.gxm) * (info.gym);
                // PetscPrintf(PETSC_COMM_WORLD, "row = %d\n", row);

                // if(i < step_size){
                //     PetscScalar value1 = 0.8;
                //     VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value1, INSERT_VALUES);
                // } else if(i < (step_size * 2)){
                //      PetscScalar value2 = 0.6;
                //      VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value2, INSERT_VALUES);
                // } else if(i < (step_size * 4)){
                //      PetscScalar value3 = 0.5;
                //      VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value3, INSERT_VALUES);
                // }else if(i < (step_size * 6)){
                //      PetscScalar value3 = 0.4;
                //      VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value3, INSERT_VALUES);
                // }else if(i < (step_size * 8)){
                //      PetscScalar value3 = 0.3;
                //      VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value3, INSERT_VALUES);
                // } else {
                //      PetscScalar value4 = 0.2;
                //      VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value4, INSERT_VALUES);
                // }

                if(i < (step_size * 3)){
                    PetscScalar value1 = 0.5;
                    VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value1, INSERT_VALUES);
                } if(i < (step_size * 6)){
                    PetscScalar value1 = 0.2;
                    VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value1, INSERT_VALUES);
                }  else {
                     PetscScalar value4 = 0.05;
                     VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value4, INSERT_VALUES);
                }

                // if(i < (step_size * 5)){
                //     PetscScalar value1 = 0.5;
                //     VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value1, INSERT_VALUES);
                // }  else {
                //      PetscScalar value4 = 0.08;
                //      VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value4, INSERT_VALUES);
                // }

                // PetscScalar value1 = 0.15;
                // VecSetValuesLocal(gradedPorousAlpha, 1, &row, &value1, INSERT_VALUES);
                
            }
        }
    }

    ierr = VecAssemblyBegin(gradedPorousAlpha);
    CHKERRQ(ierr);
    ierr = VecAssemblyEnd(gradedPorousAlpha);
    CHKERRQ(ierr);


    // end of creating alpha

    if (robustStatus) {
        DMCreateGlobalVector(da_elem,&xPhysEro);
        DMCreateGlobalVector(da_elem,&xPhysDil);
    }

    // Total number of design variables
    VecGetSize(xPhys, &n);

    PetscBool flg;

    // Optimization paramteres
    PetscOptionsGetReal(NULL, NULL, "-Emin", &Emin, &flg);
    PetscOptionsGetReal(NULL, NULL, "-Emax", &Emax, &flg);
    PetscOptionsGetReal(NULL, NULL, "-nu", &nu, &flg);
    PetscOptionsGetReal(NULL, NULL, "-volfrac", &volfrac, &flg);
    PetscOptionsGetReal(NULL, NULL, "-penal", &penal, &flg);
    PetscOptionsGetReal(NULL, NULL, "-rmin", &rmin, &flg);
    PetscOptionsGetInt(NULL, NULL, "-maxItr", &maxItr, &flg);
    PetscOptionsGetInt(NULL, NULL, "-filter", &filter, &flg);
    PetscOptionsGetReal(NULL, NULL, "-Xmin", &Xmin, &flg);
    PetscOptionsGetReal(NULL, NULL, "-Xmax", &Xmax, &flg);
    PetscOptionsGetReal(NULL, NULL, "-movlim", &movlim, &flg);
    PetscOptionsGetBool(NULL, NULL, "-projectionFilter", &projectionFilter, &flg);
    PetscOptionsGetReal(NULL, NULL, "-beta", &beta, &flg);
    PetscOptionsGetReal(NULL, NULL, "-betaFinal", &betaFinal, &flg);
    PetscOptionsGetReal(NULL, NULL, "-eta", &eta, &flg);

    PetscPrintf(PETSC_COMM_WORLD, "################### Optimization settings ####################\n");
    PetscPrintf(PETSC_COMM_WORLD, "# Problem size: n= %i, m= %i\n", n, m);
    PetscPrintf(PETSC_COMM_WORLD, "# -filter: %i  (0=sens., 1=dens, 2=PDE)\n", filter);
    PetscPrintf(PETSC_COMM_WORLD, "# -rmin: %f\n", rmin);
    PetscPrintf(PETSC_COMM_WORLD, "# -projectionFilter: %i  (0/1)\n", projectionFilter);
    PetscPrintf(PETSC_COMM_WORLD, "# -localVolume: %i  (0/1)\n", localVolumeStatus);
    PetscPrintf(PETSC_COMM_WORLD, "# -RobustApproach: %i  (0/1)\n", robustStatus);
    PetscPrintf(PETSC_COMM_WORLD, "# -User defined objective/constraint: %i  (0/1)\n", data.objectiveInput);
    PetscPrintf(PETSC_COMM_WORLD, "# -beta: %f\n", beta);
    PetscPrintf(PETSC_COMM_WORLD, "# -betaFinal: %f\n", betaFinal);
    PetscPrintf(PETSC_COMM_WORLD, "# -eta: %f\n", eta);
    PetscPrintf(PETSC_COMM_WORLD, "# -volfrac: %f\n", volfrac);
    PetscPrintf(PETSC_COMM_WORLD, "# -penal: %f\n", penal);
    PetscPrintf(PETSC_COMM_WORLD, "# -Emin/-Emax: %e - %e \n", Emin, Emax);
    PetscPrintf(PETSC_COMM_WORLD, "# -nu: %f \n", nu);
    PetscPrintf(PETSC_COMM_WORLD, "# -maxItr: %i\n", maxItr);
    PetscPrintf(PETSC_COMM_WORLD, "# -movlim: %f\n", movlim);
    PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");

    // Allocate after input
    gx = new PetscScalar[m];
    if (filter == 0) {
        Xmin = 0.001; // Prevent division by zero in filter
    }

    // Allocate the optimization vectors
    ierr = VecDuplicate(xPhys, &x);
    CHKERRQ(ierr);
    ierr = VecDuplicate(xPhys, &xTilde);
    CHKERRQ(ierr);

    VecSet(x, volfrac);      // Initialize to volfrac !
    VecSet(xTilde, volfrac); // Initialize to volfrac !
    VecSet(xPhys, volfrac);  // Initialize to volfrac !

    if (robustStatus) {
        VecSet(xPhysEro, volfrac);
	    VecSet(xPhysDil, volfrac);
    }

    // Sensitivity vectors
    ierr = VecDuplicate(x, &dfdx);
    CHKERRQ(ierr);
    ierr = VecDuplicateVecs(x, m, &dgdx);
    CHKERRQ(ierr);

    // Bounds and
    VecDuplicate(x, &xmin);
    VecDuplicate(x, &xmax);
    VecDuplicate(x, &xold);
    VecSet(xold, volfrac);

    // create xPassive vector
    ierr = VecDuplicate(xPhys, &xPassive);
    CHKERRQ(ierr);

    // Onlz if a custom domain is given
    if (xPassiveStatus) {

        // total number of elements on core
        PetscInt nel;
        VecGetLocalSize(xPhys, &nel);

        VecSet(x, 0.0);
        VecSet(xTilde, 0.0);
        VecSet(xPhys, 0.0);
        VecSet(xold, 0.0);

        // create mapping vector
        ierr = VecDuplicate(xPhys, &indicator);
        CHKERRQ(ierr);

        // counters for total and active elements on this processor
        PetscInt tcount = 0; // total number of elements
        PetscInt acount = 0; // number of active elements
        PetscInt scount = 0; // number of solid elements
        PetscInt rcount = 0; // number of rigid element
        PetscInt vcount = 0; // number of void element

        // create a temporary vector with natural ordering to get domain data to xPassive
        Vec TMPnatural;
        DMDACreateNaturalVector(da_elem, &TMPnatural);

        PetscInt low;
        PetscInt high;
        VecGetOwnershipRange(TMPnatural, &low, &high);

        PetscScalar *xnatural;
        VecGetArray(TMPnatural, &xnatural);

        // fill the temporary vector with natural ordering
        for (PetscInt i = low; i < high; i++) {
            xnatural[i-low] = data.xPassive_w[i];
        }

        VecRestoreArray(TMPnatural, &xnatural);

        // convert from natural to PETSc ordering in xPassive
        DMDANaturalToGlobalBegin(da_elem, TMPnatural, INSERT_VALUES, xPassive);
        DMDANaturalToGlobalEnd(da_elem, TMPnatural, INSERT_VALUES, xPassive);
        VecDestroy(&TMPnatural);

        // index set for xPassive and indicator
        PetscScalar *xpPassive;
        PetscScalar *xpIndicator;
        ierr = VecGetArray(xPassive, &xpPassive);
        CHKERRQ(ierr);
        ierr = VecGetArray(indicator, &xpIndicator);
        CHKERRQ(ierr);

        // count the number of active, solid and rigid elements
        // active is -1, passive is 1, solid is 2, rigid is 3
        for (PetscInt el = 0; el < nel; el++) {
            if (xpPassive[el] == -1.0) {
                acount++;
            }
            if (xpPassive[el] == 2.0) {
                scount++;
            }
            if (xpPassive[el] == 3.0) {
                rcount++;
            }
            if (xpPassive[el] == 4.0) {
                vcount++;
            }
        }

        // Forcing PETSc ordering in stead of natural ordering otherwise the output is changed into natural ordering
        //PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_NATIVE);

        // printing
        //PetscPrintf(PETSC_COMM_SELF, "tcount: %i\n", tcount);
        //PetscPrintf(PETSC_COMM_SELF, "acount: %i\n", acount);
        //PetscPrintf(PETSC_COMM_SELF, "scount: %i\n", scount);
        //PetscPrintf(PETSC_COMM_SELF, "rcount: %i\n", rcount);

        // Allreduce, number of active elements
        // tmp number of var on proces
        // acount total number of var sumed
        PetscInt tmp = acount;
        acount = 0;
        MPI_Allreduce(&tmp, &acount, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

        // Allreduce, number of solid elements
        // tmp number of var on proces
        // acount total number of var sumed
        PetscInt tmps = scount;
        scount = 0;
        MPI_Allreduce(&tmps, &scount, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

        // Allreduce, number of rigid elements
        // tmp number of var on proces
        // acount total number of var sumed
        PetscInt tmpr = rcount;
        rcount = 0;
        MPI_Allreduce(&tmpr, &rcount, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

        // Allreduce, number of void elements
        // tmp number of var on proces
        // acount total number of var sumed
        PetscInt tmpv = vcount;
        vcount = 0;
        MPI_Allreduce(&tmpv, &vcount, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

        // printing
        PetscPrintf(PETSC_COMM_WORLD, "################### STL readin ###############################\n");
        PetscPrintf(PETSC_COMM_WORLD, "acount sum: %i\n", acount);
        PetscPrintf(PETSC_COMM_WORLD, "scount sum: %i\n", scount);
        PetscPrintf(PETSC_COMM_WORLD, "rcount sum: %i\n", rcount);
        PetscPrintf(PETSC_COMM_WORLD, "vcount sum: %i\n", vcount);
        PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");


        //// create MMA vectors
        VecCreateMPI(PETSC_COMM_WORLD, tmp, acount, &xMMA);

        //VecSet(xMMA, volfrac);

        PetscScalar *xpPhys, *xptilde, *xpx, *xpold;
        VecGetArray(xPhys, &xpPhys);
        VecGetArray(xTilde, &xptilde);
        VecGetArray(x, &xpx);
        VecGetArray(xold, &xpold);

        // fill the indicator vector as mapping for x -> xMMA
        PetscInt count = 0;
        for (PetscInt el = 0; el < nel; el++) {
            if (xpPassive[el] == -1.0) {
                xpIndicator[count] = el;
                xpPhys[el] = volfrac;
                xptilde[el] = volfrac;
                xpx[el] = volfrac;
                xpold[el] = volfrac;
                count++;
            }
        }

        VecRestoreArray(xPhys, &xpPhys);
        VecRestoreArray(xTilde, &xptilde);
        VecRestoreArray(x, &xpx);
        VecRestoreArray(xold, &xpold);

        // restore xPassive, indicator
        ierr = VecRestoreArray(xPassive, &xpPassive);
        CHKERRQ(ierr);
        ierr = VecRestoreArray(indicator, &xpIndicator);
        CHKERRQ(ierr);

        // check xPassive, indicator
        //VecView(xPassive, PETSC_VIEWER_STDOUT_WORLD);
        //VecView(xMMA, PETSC_VIEWER_STDOUT_WORLD);
        //VecView(xPhys, PETSC_VIEWER_STDOUT_WORLD);
        //VecView(indicator, PETSC_VIEWER_STDOUT_WORLD);

        // duplicate needed vectors
        VecDuplicate(xMMA, &dfdxMMA);
        VecDuplicateVecs(xMMA, m, &dgdxMMA);
        VecDuplicate(xMMA, &xminMMA);
        VecDuplicate(xMMA, &xmaxMMA);
        VecDuplicate(xMMA, &xoldMMA);

    } else {
        VecSet(xPassive, -1.0);
    }

    return (ierr);
}

PetscErrorCode TopOpt::AllocateMMAwithRestart(PetscInt* itr, MMA** mma) {

    PetscErrorCode ierr = 0;

    // Set MMA parameters (for multiple load cases)? for multiple constraints
    PetscScalar aMMA[m];
    PetscScalar cMMA[m];
    PetscScalar dMMA[m];
    for (PetscInt i = 0; i < m; i++) {
        aMMA[i] = 0.0;
        dMMA[i] = 0.0;
        cMMA[i] = 1000.0;
    }

    // Check if restart is desired
    restart                  = PETSC_TRUE;  // DEFAULT USES RESTART
    flip                     = PETSC_TRUE;  // BOOL to ensure that two dump streams are kept
    PetscBool onlyLoadDesign = PETSC_FALSE; // Default restarts everything

    // Get inputs
    PetscBool flg;
    char      filenameChar[PETSC_MAX_PATH_LEN];
    PetscOptionsGetBool(NULL, NULL, "-restart", &restart, &flg);
    PetscOptionsGetBool(NULL, NULL, "-onlyLoadDesign", &onlyLoadDesign, &flg);

    if (restart) {
        ierr = VecDuplicate(x, &xo1);
        CHKERRQ(ierr);
        ierr = VecDuplicate(x, &xo2);
        CHKERRQ(ierr);
        ierr = VecDuplicate(x, &U);
        CHKERRQ(ierr);
        ierr = VecDuplicate(x, &L);
        CHKERRQ(ierr);
    }

    // Determine the right place to write the new restart files
    std::string filenameWorkdir = "./";
    PetscOptionsGetString(NULL, NULL, "-workdir", filenameChar, sizeof(filenameChar), &flg);
    if (flg) {
        filenameWorkdir = "";
        filenameWorkdir.append(filenameChar);
    }
    filename00    = filenameWorkdir;
    filename00Itr = filenameWorkdir;
    filename01    = filenameWorkdir;
    filename01Itr = filenameWorkdir;

    filename00.append("/Restart00.dat");
    filename00Itr.append("/Restart00_itr_f0.dat");
    filename01.append("/Restart01.dat");
    filename01Itr.append("/Restart01_itr_f0.dat");

    // Where to read the restart point from
    std::string restartFileVec = ""; // NO RESTART FILE !!!!!
    std::string restartFileItr = ""; // NO RESTART FILE !!!!!

    PetscOptionsGetString(NULL, NULL, "-restartFileVec", filenameChar, sizeof(filenameChar), &flg);
    if (flg) {
        restartFileVec.append(filenameChar);
    }
    PetscOptionsGetString(NULL, NULL, "-restartFileItr", filenameChar, sizeof(filenameChar), &flg);
    if (flg) {
        restartFileItr.append(filenameChar);
    }

    // Which solution to use for restarting
    PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");
    PetscPrintf(PETSC_COMM_WORLD, "# Continue from previous iteration (-restart): %i \n", restart);
    PetscPrintf(PETSC_COMM_WORLD, "# Restart file (-restartFileVec): %s \n", restartFileVec.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "# Restart file (-restartFileItr): %s \n", restartFileItr.c_str());
    PetscPrintf(PETSC_COMM_WORLD,
                "# New restart files are written to (-workdir): %s "
                "(Restart0x.dat and Restart0x_itr_f0.dat) \n",
                filenameWorkdir.c_str());

    // Check if files exist:
    PetscBool vecFile = fexists(restartFileVec);
    if (!vecFile) {
        PetscPrintf(PETSC_COMM_WORLD, "File: %s NOT FOUND \n", restartFileVec.c_str());
    }
    PetscBool itrFile = fexists(restartFileItr);
    if (!itrFile) {
        PetscPrintf(PETSC_COMM_WORLD, "File: %s NOT FOUND \n", restartFileItr.c_str());
    }

    // Read from restart point
    PetscInt nGlobalDesignVar;
    VecGetSize(x, &nGlobalDesignVar); // ASSUMES THAT SIZE IS ALWAYS MATCHED TO CURRENT MESH
    if (restart && vecFile && itrFile) {

        PetscViewer view;
        // Open the data files
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, restartFileVec.c_str(), FILE_MODE_READ, &view);

        VecLoad(x, view);
        VecLoad(xPhys, view);
        VecLoad(xo1, view);
        VecLoad(xo2, view);
        VecLoad(U, view);
        VecLoad(L, view);
        PetscViewerDestroy(&view);

        // Read iteration and fscale
        std::fstream itrfile(restartFileItr.c_str(), std::ios_base::in);
        itrfile >> itr[0];
        itrfile >> fscale;

        // Choose if restart is full or just an initial design guess
        if (onlyLoadDesign) {
            PetscPrintf(PETSC_COMM_WORLD, "# Loading design from file: %s \n", restartFileVec.c_str());
            *mma = new MMA(nGlobalDesignVar, m, x, aMMA, cMMA, dMMA);
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "# Continue optimization from file: %s \n", restartFileVec.c_str());
            *mma = new MMA(nGlobalDesignVar, m, *itr, xo1, xo2, U, L, aMMA, cMMA, dMMA);
        }
        PetscPrintf(PETSC_COMM_WORLD, "# Successful restart from file: %s and %s \n", restartFileVec.c_str(),
                    restartFileItr.c_str());
    } else if (xPassiveStatus) {
        PetscInt nxMMA;
        VecGetSize(xMMA, &nxMMA);
        *mma = new MMA(nxMMA, m, xMMA, aMMA, cMMA, dMMA);
        //PetscPrintf(PETSC_COMM_WORLD, "MMA create, just checking\n");
    } else {
        *mma = new MMA(nGlobalDesignVar, m, x, aMMA, cMMA, dMMA);
    }

    return ierr;
}

PetscErrorCode TopOpt::WriteRestartFiles(PetscInt* itr, MMA* mma) {

    PetscErrorCode ierr = 0;
    // Only dump data if correct allocater has been used
    if (!restart) {
        return -1;
    }

    // Get restart vectors
    mma->Restart(xo1, xo2, U, L);

    // Choose previous set of restart files
    if (flip) {
        flip = PETSC_FALSE;
    } else {
        flip = PETSC_TRUE;
    }

    // Write file with iteration number of f0 scaling
    // and a file with the MMA-required vectors, in the following order:
    // : x,xPhys,xold1,xold2,U,L
    PetscViewer view;         // vectors
    PetscViewer restartItrF0; // scalars

    PetscViewerCreate(PETSC_COMM_WORLD, &restartItrF0);
    PetscViewerSetType(restartItrF0, PETSCVIEWERASCII);
    PetscViewerFileSetMode(restartItrF0, FILE_MODE_WRITE);

    // Open viewers for writing
    if (!flip) {
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename00.c_str(), FILE_MODE_WRITE, &view);
        PetscViewerFileSetName(restartItrF0, filename00Itr.c_str());
    } else if (flip) {
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename01.c_str(), FILE_MODE_WRITE, &view);
        PetscViewerFileSetName(restartItrF0, filename01Itr.c_str());
    }

    // Write iteration and fscale
    PetscViewerASCIIPrintf(restartItrF0, "%d ", itr[0]);
    PetscViewerASCIIPrintf(restartItrF0, " %e", fscale);
    PetscViewerASCIIPrintf(restartItrF0, "\n");

    // Write vectors
    VecView(x, view); // the design variables
    VecView(xPhys, view);
    VecView(xo1, view);
    VecView(xo2, view);
    VecView(U, view);
    VecView(L, view);

    // Clean up
    PetscViewerDestroy(&view);
    PetscViewerDestroy(&restartItrF0);

    return ierr;
}

PetscErrorCode TopOpt::UpdateVariables(PetscInt updateDirection, Vec elementVector, Vec MMAVector) {
    // If updateDirection > 0, MMAVector is updated from fullVector
    // If updateDirection < 0, fullVector is updated from MMAVector
    PetscErrorCode ierr;
    // Pointers to the vectors
    PetscScalar *xp, *xpMMA, *indicesMap;
    //PetscInt indicesMap;
    ierr = VecGetArray(MMAVector, &xpMMA);
    CHKERRQ(ierr);
    ierr = VecGetArray(elementVector, &xp);
    CHKERRQ(ierr);
    // Index set
    PetscInt nLocalVar;
    VecGetLocalSize(xMMA, &nLocalVar);

    ierr = VecGetArray(indicator, &indicesMap);
    CHKERRQ(ierr);

    // Run through the indices
    for (PetscInt i = 0; i < nLocalVar; i++) {
        if (updateDirection > 0) {
            PetscInt index = indicesMap[i];
            xpMMA[i] = xp[index];
        } else if (updateDirection < 0) {
            PetscInt index = indicesMap[i];
            xp[index] = xpMMA[i];
        }
    }

    // Restore
    ierr = VecRestoreArray(elementVector, &xp);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(MMAVector, &xpMMA);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(indicator, &indicesMap);
    CHKERRQ(ierr);

    return ierr;
}

PetscErrorCode TopOpt::SetVariables(Vec xVector, Vec passiveVector) {

    PetscErrorCode ierr;

    // Add rigid, solid and void elements
    PetscScalar *xpPassive;
    PetscScalar *xp;
    ierr = VecGetArray(passiveVector, &xpPassive);
    CHKERRQ(ierr);
    ierr = VecGetArray(xVector, &xp);
    CHKERRQ(ierr);

    PetscInt nel;
    ierr = VecGetLocalSize(x, &nel);
    CHKERRQ(ierr);

    for (PetscInt el = 0; el < nel; el++) {
        if (xpPassive[el] == 3.0) {
            xp[el] = 10.0;
        }
        if (xpPassive[el] == 2.0) {
            xp[el] = 1.0;
        }
        if (xpPassive[el] == 4.0) {
            xp[el] = 0.0;
        }
    }

    ierr = VecRestoreArray(passiveVector, &xpPassive);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(xVector, &xp);
    CHKERRQ(ierr);

    return ierr;
}
