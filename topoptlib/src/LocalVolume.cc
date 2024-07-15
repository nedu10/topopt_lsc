#include <LocalVolume.h>


/* -----------------------------------------------------------------------------

Modified by: Thijs Smit, Dec 2020

Authors: Niels Aage, June 2019
Copyright (C) 2013-2019,

This LocalVolume implementation is licensed under Version 2.1 of the GNU
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


LocalVolume::LocalVolume(DM da_nodes, Vec x, Vec xPassive, Vec gradedPorousAlpha, DataObj data){
	// Set all pointers to NULL
	H=NULL;
	Hs=NULL;
        xVol=NULL;
	da_elem=NULL;
        R = data.Rlocvol_w;
        pnorm = 16.0;
        alpha = data.alpha_w;
        naEl = data.nael;

        // Create the xVol vector
        VecDuplicate(x,&xVol);

	// Call the setup method
	SetUp(da_nodes, xPassive);
}

LocalVolume::~LocalVolume(){
	// Deallocate data
	if (xVol!=NULL){ VecDestroy(&xVol); }
        if (Hs!=NULL){ VecDestroy(&Hs); }
	if (H!=NULL){ MatDestroy(&H); }
	if (da_elem!=NULL){ DMDestroy(&da_elem); }

}

// LocalVolume design variables
PetscErrorCode LocalVolume::Constraint(Vec x, PetscScalar *gx, Vec dx, Vec gradedPorousAlpha){
	PetscErrorCode ierr;

	// LocalVolume the design variables or copy to xPhys
	// ONLY STANDARD NEIGHBORHOOD
        ierr = MatMult(H,x,xVol); 
        CHKERRQ(ierr);
	VecPointwiseDivide(xVol,xVol,Hs);

	// Compute the p-norm function: g = (1/(n)*sum(xVol.^pnorm))^(1/pnorm)/alpha - 1.0;
        PetscScalar *xv, *alp, *dxp, gxloc = 0.0;;
        PetscInt nelloc, nelglob;
        VecGetLocalSize(xVol,&nelloc);
        //VecGetSize(xVol,&nelglob);
        nelglob = naEl; // Use number of active elements
        //PetscPrintf(PETSC_COMM_WORLD, "n: %i\n", nelglob);

        // Compute power sum
        VecGetArray(xVol,&xv);
        VecGetArray(gradedPorousAlpha,&alp);
        for (PetscInt i=0;i<nelloc;i++){

                // graded porous
            gxloc += PetscPowScalar((xv[i]/alp[i]),pnorm);

        //     gxloc += PetscPowScalar(xv[i],pnorm);
        }
        // Collect from procs
        MPI_Allreduce(&gxloc,&(gx[1]),1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

        // graded porous
        gx[1] = PetscPowScalar(gx[1] / ((PetscScalar)nelglob),1.0/pnorm) - 1.0;

        // gx[1] = PetscPowScalar(gx[1] / ((PetscScalar)nelglob),1.0/pnorm) /  alpha - 1.0;

        PetscPrintf(PETSC_COMM_WORLD, "Local volume constraint value: %f\n", gx[1]);

        // Compute the Gradients
        //dgVoldx = 1/(alpha*nelx*nely)*(1/(nelx*nely)*sum(xVol.^penal_norm))^(1/penal_norm-1)*xVol.^(penal_norm-1);

        // Compute right most term elelemtn by element
        VecSet(dx,0.0); // Un-necessary, but still...
        VecGetArray(dx,&dxp);
        for (PetscInt i=0;i<nelloc;i++){
                // graded porous
            dxp[i] = PetscPowScalar(xv[i],pnorm-1.0) / PetscPowScalar(alp[i],pnorm);

        //     dxp[i] = PetscPowScalar(xv[i],pnorm-1.0);
        }

        // Compute pre-term summed
        PetscScalar dxfact = 0.0; // Reuse x^p sum from gxloc
        MPI_Allreduce(&gxloc,&dxfact,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);

        //graded porous structure
        dxfact = 1.0/((PetscScalar)nelglob) * PetscPowScalar(dxfact / ((PetscScalar)nelglob),1.0/pnorm - 1.0);

        // dxfact = 1.0/(alpha*((PetscScalar)nelglob)) * PetscPowScalar(dxfact / ((PetscScalar)nelglob),1.0/pnorm - 1.0);

        // Scale the derivative
        VecScale(dx,dxfact);

        // restore arrays
        VecRestoreArray(xVol,&xv);
        VecRestoreArray(dx,&dxp);

        //graded porous
        VecRestoreArray(gradedPorousAlpha,&alp);

        // Update the for average operator
        Vec xtmp;
        ierr = VecDuplicate(xVol,&xtmp);  CHKERRQ(ierr);
        // dx
        VecPointwiseDivide(xtmp,dx,Hs);
        MatMult(H,xtmp,dx);

        // tidy up
        VecDestroy(&xtmp);

	return ierr;
}

//PetscErrorCode LocalVolume::SetUp(TopOpt *opt){
PetscErrorCode LocalVolume::SetUp(DM da_nodes, Vec xPassive){

	PetscErrorCode ierr;
        PetscBool flg;
        PetscOptionsGetReal(NULL,NULL,"-alpha",&alpha,&flg);
        PetscOptionsGetReal(NULL,NULL,"-pnorm",&pnorm,&flg);
        PetscOptionsGetReal(NULL,NULL,"-Rlocvol",&R,&flg);

        PetscPrintf(PETSC_COMM_WORLD,"########################################################################\n");
	PetscPrintf(PETSC_COMM_WORLD,"###################### Local volume constraint #########################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# -Rlocvol: %f \n",R);
        PetscPrintf(PETSC_COMM_WORLD,"# -alpha: %f \n",alpha);
        PetscPrintf(PETSC_COMM_WORLD,"# -pnorm: %f \n",pnorm);
        // Extract information from the nodal mesh
        PetscInt M,N,P,md,nd,pd;
        DMBoundaryType bx, by, bz;
        DMDAStencilType stype;
        ierr = DMDAGetInfo(da_nodes,NULL,&M,&N,&P,&md,&nd,&pd,NULL,NULL,&bx,&by,&bz,&stype); CHKERRQ(ierr);

        // Find the element size
        Vec lcoor;
        DMGetCoordinatesLocal(da_nodes,&lcoor);
        PetscScalar *lcoorp;
        VecGetArray(lcoor,&lcoorp);

        PetscInt nel, nen;
        const PetscInt *necon;
        DMDAGetElements_3D(da_nodes,&nel,&nen,&necon);

        PetscScalar dx,dy,dz;
        // Use the first element to compute the dx, dy, dz
        dx = lcoorp[3*necon[0*nen + 1]+0]-lcoorp[3*necon[0*nen + 0]+0];
        dy = lcoorp[3*necon[0*nen + 2]+1]-lcoorp[3*necon[0*nen + 1]+1];
        dz = lcoorp[3*necon[0*nen + 4]+2]-lcoorp[3*necon[0*nen + 0]+2];
        VecRestoreArray(lcoor,&lcoorp);

        // Create the minimum element connectivity shit
        PetscInt ElemConn;
        // Check dx,dy,dz and find max conn for a given rmin
        ElemConn = (PetscInt)PetscMax(ceil(R/dx)-1,PetscMax(ceil(R/dy)-1,ceil(R/dz)-1));
        ElemConn = PetscMin(ElemConn,PetscMin((M-1)/2,PetscMin((N-1)/2,(P-1)/2)));

        // The following is needed due to roundoff errors
        PetscInt tmp;
        MPI_Allreduce(&ElemConn, &tmp, 1,MPIU_INT, MPI_MAX,PETSC_COMM_WORLD );
        ElemConn = tmp;

        // Print to screen: mesh overlap!
        PetscPrintf(PETSC_COMM_WORLD,"# LocalVolume radius R = %f results in a stencil of %i elements \n",R,ElemConn);
        PetscPrintf(PETSC_COMM_WORLD,"########################################################################\n");

        // Find the geometric partitioning of the nodal mesh, so the element mesh will coincide
        PetscInt *Lx=new PetscInt[md];
        PetscInt *Ly=new PetscInt[nd];
        PetscInt *Lz=new PetscInt[pd];

        // get number of nodes for each partition
        const PetscInt *LxCorrect, *LyCorrect, *LzCorrect;
        DMDAGetOwnershipRanges(da_nodes, &LxCorrect, &LyCorrect, &LzCorrect);

        // subtract one from the lower left corner.
        for (int i=0; i<md; i++){
                Lx[i] = LxCorrect[i];
                if (i==0){Lx[i] = Lx[i]-1;}
        }
        for (int i=0; i<nd; i++){
                Ly[i] = LyCorrect[i];
                if (i==0){Ly[i] = Ly[i]-1;}
        }
        for (int i=0; i<pd; i++){
                Lz[i] = LzCorrect[i];
                if (i==0){Lz[i] = Lz[i]-1;}
        }

        // Create the element grid:
        DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stype,M-1,N-1,P-1,md,nd,pd,
                        1,ElemConn,Lx,Ly,Lz,&da_elem);
        // Initialize
        DMSetFromOptions(da_elem);
        DMSetUp(da_elem);

        // Set the coordinates: from 0+dx/2 to xmax-dx/2 and so on
        PetscScalar xmax = (M-1)*dx;
        PetscScalar ymax = (N-1)*dy;
        PetscScalar zmax = (P-1)*dz;
        DMDASetUniformCoordinates(da_elem , dx/2.0,xmax-dx/2.0, dy/2.0,ymax-dy/2.0, dz/2.0,zmax-dz/2.0);

        // Allocate and assemble
        DMCreateMatrix(da_elem,&H);
        DMCreateGlobalVector(da_elem,&Hs);

        // Set the filter matrix and vector
        DMGetCoordinatesLocal(da_elem,&lcoor);
        VecGetArray(lcoor,&lcoorp);
        DMDALocalInfo info;
        DMDAGetLocalInfo(da_elem,&info);
        // The variables from info that are used are described below:
        // -------------------------------------------------------------------------
        // sw = Stencil width
        // mx, my, mz = Global number of "elements" in each direction
        // xs, ys, zs = Starting point of this processor, excluding ghosts
        // xm, ym, zm = Number of grid points on this processor, excluding ghosts
        // gxs, gys, gzs = Starting point of this processor, including ghosts
        // gxm, gym, gzm = Number of grid points on this processor, including ghosts
        // -------------------------------------------------------------------------

        // Outer loop is local part = find row
        // What is done here, is:
        //
        // 1. Run through all elements in the mesh - should not include ghosts
        for (PetscInt k=info.zs; k<info.zs+info.zm; k++) {
                for (PetscInt j=info.ys; j<info.ys+info.ym; j++) {
                        for (PetscInt i=info.xs; i<info.xs+info.xm; i++) {
                                // The row number of the element we are considering:
                                PetscInt row = (i-info.gxs) + (j-info.gys)*(info.gxm) + (k-info.gzs)*(info.gxm)*(info.gym);
                                //
                                // 2. Loop over nodes (including ghosts) within a cubic domain with center at (i,j,k)
                                //    For each element, run through all elements in a box of size stencilWidth * stencilWidth * stencilWidth
                                //    Remark, we want to make sure we are not running "out of the domain",
                                //    therefore k2 etc. are limited to the max global index (info.mz-1 etc.)
                                for (PetscInt k2=PetscMax(k-info.sw,0);k2<=PetscMin(k+info.sw,info.mz-1);k2++){
                                        for (PetscInt j2=PetscMax(j-info.sw,0);j2<=PetscMin(j+info.sw,info.my-1);j2++){
                                                for (PetscInt i2=PetscMax(i-info.sw,0);i2<=PetscMin(i+info.sw,info.mx-1);i2++){
                                                        PetscInt col = (i2-info.gxs) + (j2-info.gys)*(info.gxm) + (k2-info.gzs)*(info.gxm)*(info.gym);
                                                        PetscScalar dist = 0.0;
                                                        // Compute the distance from the "col"-element to the "row"-element
                                                        for(PetscInt kk=0; kk<3; kk++){
                                                                dist = dist + PetscPowScalar(lcoorp[3*row+kk]-lcoorp[3*col+kk],2.0);
                                                        }
                                                        dist = PetscSqrtScalar(dist);
                                                        if (dist<R){
                                                                // Equal weight average
                                                                dist = 1.0;
                                                                MatSetValuesLocal(H, 1, &row, 1, &col, &dist, INSERT_VALUES);
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
        // Assemble H:
        MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

        ///////////////////////////////////////////////////////////////////////////////////////////////////

        // FOR PASSIVE
        DMDAGetElements_3D(da_nodes, &nel, &nen, &necon);

        // ELimination vector to kill the unwanted dofs in the filter matrix
        Vec Nvec;
        DMCreateGlobalVector(da_elem, &Nvec);

        // Map the xPssize vector to a new one with only 0 (passive) and 1 (active)
        Vec tmpV;
        VecDuplicate(xPassive, &tmpV);

        PetscScalar *xpPassive, *ptmp;
        ierr = VecGetArray(xPassive, &xpPassive);
        CHKERRQ(ierr);
        ierr = VecGetArray(tmpV, &ptmp);
        CHKERRQ(ierr);

        // Loop over elements and write to tmp vector
        for (PetscInt el = 0; el < nel; el++) {
            if (xpPassive[el] > 0) { // Is passive element
                ptmp[el] = 0.0;
            } else {
                ptmp[el] = 1.0;
            }
        }

        // Restore
        ierr = VecRestoreArray(xPassive, &xpPassive);
        CHKERRQ(ierr);
        ierr = VecRestoreArray(tmpV, &ptmp);
        CHKERRQ(ierr);

        // Transfer results to Nvec
        VecCopy(tmpV, Nvec);

        // Impose the dirichlet conditions, i.e. K = N'*K*N - (N-I)
        // 1.: K = N'*K*N
        MatDiagonalScale(H, Nvec, Nvec);

        // Compute the Hs, i.e. sum the rows
        Vec dummy;
        VecDuplicate(Hs, &dummy);
        VecSet(dummy, 1.0);
        MatMult(H, dummy, Hs);

        // Insert ones at zero positions on diagonal
        Vec NI;
        VecDuplicate(Nvec, &NI);
        VecSet(NI, 1.0);
        VecAXPY(NI, -1.0, Nvec);
        MatDiagonalSet(H, NI, ADD_VALUES);
        // Add ones to the HS vector to avoid division by zero
        VecAXPY(Hs, 1.0, NI);
        VecDestroy(&NI);
        ///////////////////////////////////////////////////////////////////////////////////////////////////

        // Clean up
        VecRestoreArray(lcoor,&lcoorp);
        VecDestroy(&dummy);
        VecDestroy(&tmpV);
        VecDestroy(&Nvec);
        delete [] Lx;
        delete [] Ly;
        delete [] Lz;



	return ierr;

}


PetscErrorCode LocalVolume::DMDAGetElements_3D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]) {
	PetscErrorCode ierr;
	DM_DA          *da = (DM_DA*)dm->data;
	PetscInt       i,xs,xe,Xs,Xe;
	PetscInt       j,ys,ye,Ys,Ye;
	PetscInt       k,zs,ze,Zs,Ze;
	PetscInt       cnt=0, cell[8], ns=1, nn=8;
	PetscInt       c;
	if (!da->e) {
		if (da->elementtype == DMDA_ELEMENT_Q1) {ns=1; nn=8;}
		ierr = DMDAGetCorners(dm,&xs,&ys,&zs,&xe,&ye,&ze);
		CHKERRQ(ierr);
		ierr = DMDAGetGhostCorners(dm,&Xs,&Ys,&Zs,&Xe,&Ye,&Ze);
		CHKERRQ(ierr);
		xe    += xs; Xe += Xs; if (xs != Xs) xs -= 1;
		ye    += ys; Ye += Ys; if (ys != Ys) ys -= 1;
		ze    += zs; Ze += Zs; if (zs != Zs) zs -= 1;
		da->ne = ns*(xe - xs - 1)*(ye - ys - 1)*(ze - zs - 1);
		PetscMalloc((1 + nn*da->ne)*sizeof(PetscInt),&da->e);
		for (k=zs; k<ze-1; k++) {
			for (j=ys; j<ye-1; j++) {
				for (i=xs; i<xe-1; i++) {
					cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[2] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[3] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs  )*(Xe-Xs)*(Ye-Ys);
					cell[4] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					cell[5] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					cell[6] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					cell[7] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) + (k-Zs+1)*(Xe-Xs)*(Ye-Ys);
					if (da->elementtype == DMDA_ELEMENT_Q1) {
						for (c=0; c<ns*nn; c++) da->e[cnt++] = cell[c];
					}
				}
			}
		}
	}
	*nel = da->ne;
	*nen = nn;
	*e   = da->e;
	return(0);
}
