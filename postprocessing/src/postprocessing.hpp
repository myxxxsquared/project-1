
#ifndef POSTPROCESSING_HEADER
#define POSTPROCESSING_HEADER

#include <vector>
#include <queue>
#include <algorithm>
using namespace std;

#include "inferencemap.hpp"
#include "config.hpp"

struct RegionInfo
{
    Mat region;
    float avg_r, avg_x, avg_y, avg_cos, avg_sin;
    int ptnumber, area;
    vector<vector<Point>> contours;
};

class PostProcessor
{
  public:
    ProcessConfig config;
    void *inferencemap;
    Mat search_mark;
    Mat result;
    Mat trmap;
    int width;
    int height;

    vector<RegionInfo> regions;

    bool postprocess_pixellink();
    bool postprocess_tcl();

    Point2i topoint(int i);
    int toint(Point2i pt);

    static vector<Point2i> generate_random_vector(int rows, int cols);
    void search_contour(Point2i pt);
};

#endif
