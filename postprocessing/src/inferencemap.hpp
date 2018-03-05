#ifndef INFERENCEMAP_HEADER
#define INFERENCEMAP_HEADER

#include "pythonheader.hpp"

struct InferencePixel
{
    float tcl, radius, cos, sin, tr;
};

template <typename PIXEL_T>
class InferenceMap
{
  public:
    int width, height;
    PyArrayObject *arrobj;
    const PIXEL_T *data;

    InferenceMap();
    InferenceMap(InferenceMap<PIXEL_T> &obj);
    ~InferenceMap();
    InferenceMap<PIXEL_T> &operator=(InferenceMap<PIXEL_T> &obj);
    bool init(PyObject *obj);

    const inline PIXEL_T &at(int x, int y) const
    {
        return data[y * width + x];
    }

    const inline PIXEL_T &at(Point2i pt) const
    {
        return this->at(pt.x, pt.y);
    }

    // inline void savetest() const
    // {
    //     Mat mat;
    //     mat.create(height, width, CV_8UC1);
    //     for (int i = 0; i < height; ++i)
    //         for (int j = 0; j < width; ++j)
    //             mat.at<uchar>(i, j) = at(j, i).tcl > 0.5 ? 255 : 0;
    //     imwrite("test.png", mat);
    // }
};

#endif /* INFERENCEMAP_HEADER */