
#define MYLIBRARY_USE_IMPORT

#include "pythonheader.hpp"
#include "postprocessing.hpp"

#include <ctime>
#include <cstdlib>

static PyObject *contour_to_obj(vector<Point> &contour)
{
    npy_intp sizes[] = {(npy_intp)contour.size(), 1, 2};
    PyObject *obj = PyArray_SimpleNew(3, sizes, NPY_INT);
    int *data = (int *)PyArray_DATA((PyArrayObject *)obj);

    int index = 0;
    for (auto pt : contour)
    {
        data[index++] = pt.x;
        data[index++] = pt.y;
    }

    return obj;
}

static PyObject *py_postprocessing(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject *pymaps = NULL, *pyconfig = NULL;

    char txt_maps[] = "maps";
    char txt_configs[] = "configs";
    char *keywords[] = {txt_maps, txt_configs, NULL};
    const char *paramformat = "OO;";

    PostProcessor processor;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-arith"
    if (NULL == PyArg_ParseTupleAndKeywords(
                    args, kw, paramformat, keywords,
                    &pymaps, &pyconfig))
        return NULL;
#pragma GCC diagnostic pop
    if (!processor.config.load_from(pyconfig))
        return NULL;

    if(processor.config.is_pixellink)
    {
        InferenceMap<Pixel_PixelLink> map;
        if(!map.init(pymaps))
            return NULL;
        processor.inferencemap = &map;

        PyAllowThreads allowthreads;
        if (!processor.postprocess_pixellink())
        {
            failmsg("process failed.");
            return NULL;
        }
    }
    else
    {
        InferenceMap<Pixel_TCL> map;
        if(!map.init(pymaps))
            return NULL;
        processor.inferencemap = &map;

        PyAllowThreads allowthreads;
        if (!processor.postprocess_tcl())
        {
            failmsg("process failed.");
            return NULL;
        }
    }

    std::vector<PyObject *> ctns;
    for (auto &region : processor.regions)
        for (auto &ctn : region.contours)
            ctns.push_back(contour_to_obj(ctn));

    PyObject *pyresult = PyList_New(ctns.size());
    for (int i = 0; i < (int)ctns.size(); ++i)
        PyList_SetItem(pyresult, i, ctns[i]);

    // processor.map.savetest();

    return pyresult;
}

static PyMethodDef methods[] = {
    {"postprocessing", (PyCFunction)py_postprocessing, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_postprocessing",
    NULL,
    -1,
    methods};

PyMODINIT_FUNC
PyInit__postprocessing(void)
{
    std::srand ( unsigned ( std::time(0) ) );
    import_array();
    return PyModule_Create(&module);
}
