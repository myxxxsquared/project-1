
#include "config.hpp"

#define READ_FLOAT(name)                                           \
    do                                                             \
    {                                                              \
        PyObject *o = PyObject_GetAttrString(obj, #name);          \
        if (o == NULL)                                             \
            return false;                                          \
        this->name = PyFloat_AsDouble(o);                          \
        Py_XDECREF(o);                                             \
        /*fprintf(stderr, "%s: %f\n", #name, (float)this->name);*/ \
    } while (0)

#define READ_INT(name)                                           \
    do                                                           \
    {                                                            \
        PyObject *o = PyObject_GetAttrString(obj, #name);        \
        if (o == NULL)                                           \
            return false;                                        \
        this->name = PyLong_AsLong(o);                           \
        Py_XDECREF(o);                                           \
        /*fprintf(stderr, "%s: %d\n", #name, (int)this->name);*/ \
    } while (0)

#define READ_BOOL(name)                                          \
    do                                                           \
    {                                                            \
        PyObject *o = PyObject_GetAttrString(obj, #name);        \
        if (o == NULL)                                           \
            return false;                                        \
        this->name = o == Py_True;                               \
        Py_XDECREF(o);                                           \
        /*fprintf(stderr, "%s: %d\n", #name, (int)this->name);*/ \
    } while (0)

bool ProcessConfig::load_from(PyObject *obj)
{
    READ_FLOAT(t_tcl);
    READ_FLOAT(t_tr);
    READ_FLOAT(t_delta);
    READ_FLOAT(t_rad);
    READ_FLOAT(fewest_tcl_ratio);
    READ_FLOAT(smallest_area_ratio);
    READ_FLOAT(smallest_area_ratio_tcl);
    READ_FLOAT(radius_scaling);
    READ_INT(fewest_tcl);
    READ_BOOL(is_pixellink);

    return true;
}
