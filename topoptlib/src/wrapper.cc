#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "structmember.h"
#include "topoptlib.h"
#include <vector>
#include <stdio.h>
#include <IO.h>
#include <cstdlib>
#include <iostream>
#include <string>

/*
Author: Thijs Smit, Dec 2020
Copyright (C) 2020 ETH Zurich

Disclaimer:
 The authors reserves all rights but does not guaranty that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/

// Initialize standard problem settings, beam example
static int Data_init(DataObj *self, PyObject *args, PyObject *kwds)
{

    // MESH
    self->xc[0] = 0.0;
    self->xc[1] = 2.0;
    self->xc[2] = 0.0;
    self->xc[3] = 1.0;
    self->xc[4] = 0.0;
    self->xc[5] = 1.0;
    //self->xc[6] = 0.0;
    //self->xc[7] = 0.0;
    //self->xc[8] = 0.0;
    //self->xc[9] = 0.0;
    //self->xc[10] = 0.0;
    //self->xc[11] = 0.0;
    //self->xc[12] = 0.0;

    self->nxyz[0] = 65; //129;
    self->nxyz[1] = 33; //65;
    self->nxyz[2] = 33; //65;

    self->nNodes = 65 * 33 * 33;
    self->nElements = 64 * 32 * 32;
    self->nDOF = self->nNodes * 3;

    // MATERIAL
    self->Emin = 1.0e-9;
    self->Emax = 1.0;
    self->nu = 0.3;
    self->penal = 3.0;

    // FILTER
    self->filter = 1;
    self->rmin = 0.08;

    // MMA
    self->maxIter = 400;
    self->tol = 0.01;

    // BC
    self->nL = 1;
    self->loadcases_list.resize(1);

    // BC1
    BC condition1;
    condition1.BCtype = 1;
    condition1.Checker_dof_vec.push_back(0);
    condition1.Checker_val_vec.push_back(0.0);
    condition1.Setter_dof_vec.push_back(0);
    condition1.Setter_dof_vec.push_back(1);
    condition1.Setter_dof_vec.push_back(2);
    condition1.Setter_val_vec.push_back(0.0);
    condition1.Setter_val_vec.push_back(0.0);
    condition1.Setter_val_vec.push_back(0.0);
    condition1.Para = 0;
    self->loadcases_list.at(0).push_back(condition1);

    // BC2
    BC condition2;
    condition2.BCtype = 2;
    condition2.Checker_dof_vec.push_back(0);
    condition2.Checker_val_vec.push_back(2.0);
    condition2.Checker_dof_vec.push_back(2);
    condition2.Checker_val_vec.push_back(0.0);
    condition2.Setter_dof_vec.push_back(2);
    condition2.Setter_val_vec.push_back(-0.001);
    condition2.Para = 0;
    self->loadcases_list.at(0).push_back(condition2);

    // BC3
    BC condition3;
    condition3.BCtype = 2;
    condition3.Checker_dof_vec.push_back(0);
    condition3.Checker_val_vec.push_back(2.0);
    condition3.Checker_dof_vec.push_back(1);
    condition3.Checker_val_vec.push_back(0.0);
    condition3.Checker_dof_vec.push_back(2);
    condition3.Checker_val_vec.push_back(0.0);
    condition3.Setter_dof_vec.push_back(2);
    condition3.Setter_val_vec.push_back(-0.0005);
    condition3.Para = 0;
    self->loadcases_list.at(0).push_back(condition3);

    // BC4
    BC condition4;
    condition4.BCtype = 2;
    condition4.Checker_dof_vec.push_back(0);
    condition4.Checker_val_vec.push_back(2.0);
    condition4.Checker_dof_vec.push_back(1);
    condition4.Checker_val_vec.push_back(1.0);
    condition4.Checker_dof_vec.push_back(2);
    condition4.Checker_val_vec.push_back(0.0);
    condition4.Setter_dof_vec.push_back(2);
    condition4.Setter_val_vec.push_back(-0.0005);
    condition4.Para = 0;
    self->loadcases_list.at(0).push_back(condition4);

    // volume fraction
    self->volumefrac = 0.12;

    // User defined objective and constraint
    self->objectiveInput = PETSC_FALSE;

    // OUTPUT
    self->outputIter = 20;
    self->writevtr = PETSC_TRUE;

    return 0;
}

static PyMemberDef members[] =
    { {"nNodes", T_INT, offsetof(DataObj, nNodes), 0, "nNodes docstring"},
      {"nElements", T_INT, offsetof(DataObj, nElements), 0, "nNodes docstring"},
      {"nDOF", T_INT, offsetof(DataObj, nDOF), 0, "nNodes docstring"},
      {"nael", T_INT, offsetof(DataObj, nael), 0, "nael docstring"},
      {"nrel", T_INT, offsetof(DataObj, nrel), 0, "nael docstring"},
      {"nsel", T_INT, offsetof(DataObj, nsel), 0, "nael docstring"},
      {"nvel", T_INT, offsetof(DataObj, nvel), 0, "nvel docstring"},
      {NULL}  /* Sentinel */
};

// set mesh variables
static PyObject *structuredGrid_py(DataObj *self, PyObject *args)
{
    double xc0, xc1, xc2, xc3, xc4, xc5; //, xc6, xc7, xc8, xc9, xc10, xc11;
    int nxyz0, nxyz1, nxyz2;

    //if(!PyArg_ParseTuple(args, "(dddddddddddd)(iii)", &xc0, &xc1, &xc2, &xc3, &xc4, &xc5, &xc6, &xc7, &xc8, &xc9, &xc10, &xc11, &nxyz0, &nxyz1, &nxyz2)) {
    //    return NULL;
    //}

    if(!PyArg_ParseTuple(args, "(dddddd)(iii)", &xc0, &xc1, &xc2, &xc3, &xc4, &xc5, &nxyz0, &nxyz1, &nxyz2)) {
        return NULL;
    }

    self->xc[0] = xc0;
    self->xc[1] = xc1;
    self->xc[2] = xc2;
    self->xc[3] = xc3;
    self->xc[4] = xc4;
    self->xc[5] = xc5;

    self->nxyz[0] = nxyz0;
    self->nxyz[1] = nxyz1;
    self->nxyz[2] = nxyz2;

    self->nNodes = nxyz0 * nxyz1 * nxyz2;
    self->nElements = (nxyz0 - 1) * (nxyz1 - 1) * (nxyz2 - 1);
    self->nDOF = self->nNodes * 3;

    self->writevtr = PETSC_FALSE;

    // resize the xPassive_wrapper vector and set defauld value -1 for active elements
    if (self->xPassive_w.size() <= 1) {
        self->xPassive_w.resize(self->nElements, -1);
    }

    // make sure that element counts are set
    self->updatecounts();

    Py_RETURN_NONE;
}

// stl read design domain
static PyObject *stlread_py(DataObj *self, PyObject *args)
{
    double encoding, backround;
    int treshold;
    double b0, b1, b2, b3, b4, b5;
    char *path = NULL;

    if (!PyArg_ParseTuple(args, "ddi(ddd)(ddd)s", &encoding, &backround, &treshold, &b0, &b1, &b2, &b3, &b4, &b5, &path)) {
        return NULL;
    }

    // check if STL is in binary format
    if (!checkBinarySTL(path)) {
        printf("Stl file is not in binary format\n");
        return NULL;
    }

    self->b_w[0] = b0;
    self->b_w[1] = b1;
    self->b_w[2] = b2;
    self->b_w[3] = b3;
    self->b_w[4] = b4;
    self->b_w[5] = b5;

    // Create geometry and read stl
    Geometry geo(path);

    // Box around the stl domain
    V3 gridMinCorner(b0, b1, b2);
    V3 gridMaxCorner(b3, b4, b5);
    V3 gridExtend = gridMaxCorner - gridMinCorner;

    double dx = self->xc[1] / (self->nxyz[0] - 1);
    double dy = self->xc[3] / (self->nxyz[1] - 1);
    double dz = self->xc[5] / (self->nxyz[2] - 1);

    // set grid bounding corner
    // gridNum = number of voxels
    int3 gridNum = int3(gridExtend.x / dx, gridExtend.y / dy, gridExtend.z / dz);

    // make the grid equal to the PETSc grid
    GridBox grid(gridMinCorner, dx, gridNum);

    // voxelize geo in grid
    Voxelizer vox(geo, grid);

    // get flag
    const char* flag = vox.get_flag();

    // resize the xPassive_wrapper vector and set defauld value 1 for void elements
    if (self->xPassive_w.size() <= 1) {
        self->xPassive_w.resize(self->nElements, 1);
    }

    // Point data to cell data
    int nelx = (self->nxyz[0] - 1);
    int nely = (self->nxyz[1] - 1);
    int nelz = (self->nxyz[2] - 1);

    int nox = 0;
    int noy = nelx + 1;
    int noz = (nelx + 1) * (nely + 1);

    int ecount = 0;
    //int acount = 0;

    for (int z = 0; z < nelz; z++) {
        for (int y = 0; y < nely; y++) {
            for (int x = 0; x < nelx; x++) {

                int node1 = flag[nox];
                int node2 = flag[nox + 1];
                int node3 = flag[noy + 1];
                int node4 = flag[noy];
                int node5 = flag[nox + noz];
                int node6 = flag[nox + 1 + noz];
                int node7 = flag[noy + 1 + noz];
                int node8 = flag[noy + noz];

                nox++;
                noy++;

                if ((x + 1) % nelx == 0) {
                    nox++;
                    noy++;
                }

                if ((x + 1) % nelx == 0 && (y + 1) % nely == 0) {
                    nox = (z + 1) * (nelx + 1) * (nely + 1);
                    noy = (z + 1) * ((nelx + 1) * (nely + 1)) + nelx + 1;
                }

                int sum = node1 + node2 + node3 + node4 + node5 + node6 + node7 + node8;

                if (sum >= treshold) {
                    self->xPassive_w.at(ecount) = encoding;
                } else if (backround != 0.0) {
                    self->xPassive_w.at(ecount) = backround;
                }

                ecount++;

            }
        }
    }

    self->updatecounts();

    Py_RETURN_NONE;
}

// set material variables
static PyObject *material_py(DataObj *self, PyObject *args)
{
    double Emin, Emax, nu, dens, penal;

    if(!PyArg_ParseTuple(args, "ddddd", &Emin, &Emax, &nu, &dens, &penal)) {
        return NULL;
    }

    self->Emin = Emin;
    self->Emax = Emax;
    self->nu = nu;
    self->penal = penal;

    Py_RETURN_NONE;
}

// set filter variables
static PyObject *filter_py(DataObj *self, PyObject *args)
{
    int filter;
    double rmin;

    if(!PyArg_ParseTuple(args, "id", &filter, &rmin)) {
        return NULL;
    }

    self->filter = filter;
    self->rmin = rmin;

    Py_RETURN_NONE;
}

static PyObject *continuation_py(DataObj *self, PyObject *args)
{
    double Pinitial, Pfinal, stepsize;
    int IterProg;

    if(!PyArg_ParseTuple(args, "dddi", &Pinitial, &Pfinal, &stepsize, &IterProg)) {
        return NULL;
    }

    self->continuation_w = 1;
    self->penalinitial_w = Pinitial;
    self->penalfinal_w = Pfinal;
    self->stepsize_w = stepsize;
    self->iterProg_w = IterProg;

    Py_RETURN_NONE;
}

static PyObject *projection_py(DataObj *self, PyObject *args)
{
    double betaFinal, betaInit, eta;
    if(!PyArg_ParseTuple(args, "ddd", &betaFinal, &betaInit, &eta)) {
        return NULL;
    }

    self->betaFinal_w = betaFinal;
    self->betaInit_w = betaInit;
    self->eta_w = eta;
    self->IterProgPro_w = 40;

    self->projection_w = 1;

    Py_RETURN_NONE;
}

static PyObject *robust_py(DataObj *self, PyObject *args)
{
    double betaFinal, betaInit, eta, delta;

    if(!PyArg_ParseTuple(args, "dddd", &betaFinal, &betaInit, &eta, &delta)) {
        return NULL;
    }

    self->betaFinal_w = betaFinal;
    self->betaInit_w = betaInit;
    self->eta_w = eta;
    self->IterProgPro_w = 40;
    self->delta_w = delta;

    self->robust_w = 1;

    Py_RETURN_NONE;
}

// set optimizer variables
static PyObject *mma_py(DataObj *self, PyObject *args)
{
    int maxIter;
    double tol;

    if(!PyArg_ParseTuple(args, "id", &maxIter, &tol)) {
        return NULL;
    }

    self->maxIter = maxIter;
    self->tol = tol;

    Py_RETURN_NONE;
}

static PyObject *bcpara_py(DataObj *self, PyObject *args)
{
    PyObject *pypara_func;
    if (!PyArg_ParseTuple(args, "O", &pypara_func))
        return NULL;

    // Make sure second argument is a function
    if (!PyCallable_Check(pypara_func))
        return NULL;

    self->para_func = pypara_func;
    Py_INCREF(self->para_func);

    Py_RETURN_NONE;
}

static PyObject *bc_py(DataObj *self, PyObject *args)
{
    PyObject *checker_dof;
    PyObject *checker_val;
    PyObject *setter_dof;
    PyObject *setter_val;
    //PyObject *param_func;

    int loadcase_ID;
    int BCtypes;

    int nChecker_dof;
    int nChecker_val;
    int nSetter_dof;
    int nSetter_val;

    int param_funci;

    BC condition;

    if (!PyArg_ParseTuple(args, "iiOOOOi", &loadcase_ID, &BCtypes, &checker_dof, &checker_val, &setter_dof, &setter_val, &param_funci)) {
        return NULL;
    }

    nChecker_dof = PyList_Size(checker_dof);
    nChecker_val = PyList_Size(checker_val);
    nSetter_dof = PyList_Size(setter_dof);
    nSetter_val = PyList_Size(setter_val);
    //printf("Loadcase ID: %i\n", loadcase_ID);
    //printf("Number of Checkers val: %i\n", nChecker_val);
    //printf("Number of Setters dof: %i\n", nSetter_dof);
    //printf("Number of Setters val: %i\n", nSetter_val);

    condition.BCtype = BCtypes;

    PyObject *iterator = PyObject_GetIter(checker_dof);
    PyObject *item;


    while ((item = PyIter_Next(iterator)))
        {
            int val = PyLong_AsLong(item);
            condition.Checker_dof_vec.push_back(val);
            Py_DECREF(item);
        }

    Py_DECREF(iterator);


    PyObject *iterato = PyObject_GetIter(checker_val);
    PyObject *ite;


    while ((ite = PyIter_Next(iterato)))
        {
            double va = PyFloat_AsDouble(ite);
            condition.Checker_val_vec.push_back(va);
            Py_DECREF(ite);
        }

    Py_DECREF(iterato);


    PyObject *iteratorr = PyObject_GetIter(setter_dof);
    PyObject *itemm;

    while ((itemm = PyIter_Next(iteratorr)))
        {
            int vall = PyLong_AsLong(itemm);
            condition.Setter_dof_vec.push_back(vall);
            Py_DECREF(itemm);
        }

    Py_DECREF(iteratorr);

    PyObject *iteratorrr = PyObject_GetIter(setter_val);
    PyObject *itemmm;

    while ((itemmm = PyIter_Next(iteratorrr)))
        {
            double valll = PyFloat_AsDouble(itemmm);
            condition.Setter_val_vec.push_back(valll);
            Py_DECREF(itemmm);
        }

    Py_DECREF(iteratorrr);


    condition.Para = param_funci;
    self->loadcases_list.at(loadcase_ID).push_back(condition);

    Py_RETURN_NONE;
}

static PyObject *loadcases_py(DataObj *self, PyObject *args)
{
    int n;

    if (!PyArg_ParseTuple(args, "i", &n)) {
        return NULL;
    }

    // Set number of loadcases variable
    self->nL = n;

    // Clean vector
    self->loadcases_list.clear();

    // Resize the loadcases list according to user input
    self->loadcases_list.resize(n);

    Py_RETURN_NONE;
}

static PyObject *localVolume_py(DataObj *self, PyObject *args)
{
    double Rlocvol, alpha;

    if (!PyArg_ParseTuple(args, "dd", &Rlocvol, &alpha)) {
        return NULL;
    }

    self->Rlocvol_w = Rlocvol;
    self->alpha_w = alpha;
    self->m = 2;
    self->localVolume_w = 1;

    Py_RETURN_NONE;
}

static PyObject *obj_py(DataObj *self, PyObject *args)
{
    PyObject *pyobj_func;
    if (!PyArg_ParseTuple(args, "O", &pyobj_func))
        return NULL;

    // Make sure second argument is a function
    if (!PyCallable_Check(pyobj_func))
        return NULL;

    self->obj_func = pyobj_func;
    Py_INCREF(self->obj_func);

    self->objectiveInput = PETSC_TRUE;

    Py_RETURN_NONE;
}

static PyObject *objsens_py(DataObj *self, PyObject *args)
{
    PyObject *pyobj_sens_func;
    if (!PyArg_ParseTuple(args, "O", &pyobj_sens_func))
        return NULL;

    // Make sure second argument is a function
    if (!PyCallable_Check(pyobj_sens_func))
        return NULL;

    self->obj_sens_func = pyobj_sens_func;
    Py_INCREF(self->obj_sens_func);

    Py_RETURN_NONE;
}

static PyObject *initialcondition_py(DataObj *self, PyObject *args)
{
    double init_volfrac;

    if(!PyArg_ParseTuple(args, "d", &init_volfrac)) {
        return NULL;
    }

    self->volumefrac = init_volfrac;

    Py_RETURN_NONE;
}

static PyObject *cons_py(DataObj *self, PyObject *args)
{
    PyObject *pyconst_func;
    if (!PyArg_ParseTuple(args, "O", &pyconst_func))
        return NULL;

    // Make sure second argument is a function
    if (!PyCallable_Check(pyconst_func))
        return NULL;

    self->const_func = pyconst_func;
    Py_INCREF(self->const_func);

    Py_RETURN_NONE;
}

static PyObject *conssens_py(DataObj *self, PyObject *args)
{
    PyObject *pyconst_sens_func;
    if (!PyArg_ParseTuple(args, "O", &pyconst_sens_func))
        return NULL;

    // Make sure second argument is a function
    if (!PyCallable_Check(pyconst_sens_func))
        return NULL;

    self->const_sens_func = pyconst_sens_func;
    Py_INCREF(self->const_sens_func);

    Py_RETURN_NONE;
}

static PyObject *check_py(DataObj *self, PyObject *args)
{
    PyObject *test_func;

    if (!PyArg_ParseTuple(args, "O", &test_func))
        return NULL;

    // Make sure second argument is a function
    if (!PyCallable_Check(test_func))
        return NULL;

    self->test_func = test_func;
    Py_INCREF(self->test_func);

    self->test_w = 1;

    Py_RETURN_NONE;
}

// stl read design domain
static PyObject *vtr_py(DataObj *self, PyObject *args)
{
    int output;

    if (!PyArg_ParseTuple(args, "i", &output)) {
        return NULL;
    }

    self->outputIter = output;
    self->writevtr = PETSC_TRUE;

    Py_RETURN_NONE;
}

static PyObject *solve_py(DataObj *self)
{
    // variable to store output variables
    int complete = 0;

    complete = solve(*self);

    // return "complete" signal
    return PyLong_FromLong(complete);
}

static PyMethodDef methods[] =
    { {"structuredGrid", (PyCFunction)structuredGrid_py, METH_VARARGS, "Implement structuredGrid\n"},
      {"stlread", (PyCFunction)stlread_py, METH_VARARGS, "STL read\n"},
      {"material", (PyCFunction)material_py, METH_VARARGS, "Implement material\n"},
      {"filter", (PyCFunction)filter_py, METH_VARARGS, "Implement filter\n"},
      {"continuation", (PyCFunction)continuation_py, METH_VARARGS, "Implement filter\n"},
      {"robust", (PyCFunction)robust_py, METH_VARARGS, "Implement filter\n"},
      {"projection", (PyCFunction)projection_py, METH_VARARGS, "Implement filter\n"},
      {"mma", (PyCFunction)mma_py, METH_VARARGS, "Implement mma\n"},
      {"bc", (PyCFunction)bc_py, METH_VARARGS, "Implement boundery conditions\n"},
      {"bcpara", (PyCFunction)bcpara_py, METH_VARARGS, "Implement boundery conditions\n"},
      {"loadcases", (PyCFunction)loadcases_py, METH_VARARGS, "Implement boundery conditions\n"},
      {"localVolume", (PyCFunction)localVolume_py, METH_VARARGS, "Implement boundery conditions\n"},
      {"obj", (PyCFunction)obj_py, METH_VARARGS, "Callback for Objective function\n"},
      {"objsens", (PyCFunction)objsens_py, METH_VARARGS, "Callback for Sensitivity function\n"},
      {"initialcondition", (PyCFunction)initialcondition_py, METH_VARARGS, "Callback for Sensitivity function\n"},
      {"cons", (PyCFunction)cons_py, METH_VARARGS, "Callback for Objective function\n"},
      {"conssens", (PyCFunction)conssens_py, METH_VARARGS, "Callback for Sensitivity function\n"},
      {"check", (PyCFunction)check_py, METH_VARARGS, "Callback for Sensitivity function\n"},
      {"vtr", (PyCFunction)vtr_py, METH_VARARGS, "Output vtr files\n"},
      {"solve", (PyCFunction)solve_py, METH_NOARGS, "Python bindings to solve() in topoptlib\n"},
      {NULL, NULL, 0, NULL}
    };

// check if you can skip 0
static PyTypeObject DataType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "topoptlib.Data",          /* tp_name */
    sizeof(DataObj),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Data objects",            /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    methods,                   /* tp_methods */
    members,                   /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,                         /* tp_init */
    (initproc)Data_init,         /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};

// for default: look into allocator or constructor

static struct PyModuleDef module =
    {
     PyModuleDef_HEAD_INIT,
     "topoptlib",                           /* m_name */
     "Python bindings to TopOpt_in_PETSc",  /* m_doc */
     -1,                                    /* m_size */
     NULL,                                  /* m_methods */
     NULL,                                  /* m_reload */
     NULL,                                  /* m_traverse */
     NULL,                                  /* m_clear */
     NULL,                                  /* m_free */
    };

PyMODINIT_FUNC PyInit_topoptlib(void)
{

    if (PyType_Ready(&DataType) < 0)
        return NULL;

    PyObject* m = PyModule_Create(&module);

    // for numpy arrays
    //import_array();

    // Adding Data python class with C variables to module at initialization
    Py_INCREF(&DataType);
    if (PyModule_AddObject(m, "Data", (PyObject *) &DataType) < 0) {
        Py_DECREF(&DataType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
