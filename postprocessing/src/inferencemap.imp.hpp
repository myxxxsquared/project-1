
template<typename PIXEL_T>
InferenceMap<PIXEL_T>::InferenceMap()
    : arrobj(NULL), data(NULL)
{
}

template<typename PIXEL_T>
bool InferenceMap<PIXEL_T>::init(PyObject *pyobj)
{
    py_assert(pyobj != NULL);
    py_assert(PyArray_Check(pyobj));
    PyArrayObject *obj = (PyArrayObject *)pyobj;

    py_assert(PyArray_TYPE(obj) == NPY_FLOAT);
    py_assert(PyArray_NDIM(obj) == 3);

    const npy_intp *_sizes = PyArray_DIMS(obj);
    const npy_intp *_strides = PyArray_STRIDES(obj);
    intptr_t channels = _sizes[2];

    py_assert(channels * sizeof(float) == sizeof(PIXEL_T));

    Py_XDECREF(this->arrobj);

    intptr_t elemsize = 4;
    intptr_t height = _sizes[0];
    intptr_t width = _sizes[1];
    bool needcopy = (_strides[2] != elemsize) || (_strides[1] != elemsize * channels) || (_strides[0] != elemsize * channels * width);

    if (needcopy)
        obj = PyArray_GETCONTIGUOUS(obj);
    else
        Py_INCREF(obj);

    if (obj == NULL)
        return false;

    this->arrobj = obj;
    this->data = (PIXEL_T *)PyArray_DATA(obj);
    this->height = height;
    this->width = width;

    // printf("strides: %d, %d\n", (int)((intptr_t)&at(1, 0) - (intptr_t)&at(0, 0)), (int)((intptr_t)&at(0, 1) - (intptr_t)&at(0, 0)));

    return true;
}

template<typename PIXEL_T>
InferenceMap<PIXEL_T>::~InferenceMap()
{
    Py_XDECREF(arrobj);
}

template<typename PIXEL_T>
InferenceMap<PIXEL_T>::InferenceMap(InferenceMap<PIXEL_T> &obj)
{
    this->width = obj.width;
    this->height = obj.height;
    this->arrobj = obj.arrobj;
    this->data = obj.data;

    Py_XINCREF(this->arrobj);
}

template<typename PIXEL_T>
InferenceMap<PIXEL_T> &InferenceMap<PIXEL_T>::operator=(InferenceMap<PIXEL_T> &obj)
{
    Py_XDECREF(this->arrobj);
    this->width = obj.width;
    this->height = obj.height;
    this->arrobj = obj.arrobj;
    this->data = obj.data;
    Py_XINCREF(this->arrobj);

    return *this;
}
