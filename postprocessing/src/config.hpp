
#ifndef CONFIG_HEADER
#define CONFIG_HEADER

#include "Python.h"

struct ProcessConfig
{
    float t_tcl, t_tr, t_delta, t_rad;
    float fewest_tcl_ratio, smallest_area_ratio, smallest_area_ratio_tcl;
    float radius_scaling;
    int fewest_tcl;
    bool load_from(PyObject *obj);
};

#endif
