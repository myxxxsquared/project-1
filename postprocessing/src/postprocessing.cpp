
#include "inferencemap.hpp"
#include "postprocessing.hpp"
#include "disjointset.hpp"

#include <cmath>
#include <vector>
#include <map>

vector<Point2i> PostProcessor::generate_random_vector(int rows, int cols)
{
    vector<Point2i> rowlist;
    rowlist.resize(rows * cols);
    int k = 0;
    for (int i = 0; i < cols; ++i)
        for (int j = 0; j < rows; ++j)
            rowlist.at(k++) = Point2i(i, j);
    random_shuffle(rowlist.begin(), rowlist.end());
    return rowlist;
}

void PostProcessor::search_contour(Point2i pt)
{
    InferenceMap<Pixel_TCL> &map = *(InferenceMap<Pixel_TCL> *)this->inferencemap;

    Pixel_TCL sp = map.at(pt);

    // vector<Point2i> points;
    deque<Point2i> to_search;
    // points.push_back(pt);
    float sum_radius = sp.radius;
    // float sum_x = pt.x;
    // float sum_y = pt.y;
    // float sum_cos = sp.cos;
    // float sum_sin = sp.sin;
    int ptnumber = 1;
    to_search.push_back(pt);
    search_mark.at<uchar>(pt) = 255;

    int search_distance = int(pow(sp.radius / 5.0, 1.0)) + 2;
    // printf("search_distance: %d\n", search_distance);
    // search_distance = 1;

    Mat region;
    region.create(height, width, CV_8UC1);
    region.setTo(Scalar((unsigned char)0));

    Mat tclregion;
    tclregion.create(height, width, CV_8UC1);
    tclregion.setTo(Scalar((unsigned char)0));

    while (to_search.size())
    {
        Point2i cur = to_search.front();
        to_search.pop_front();

        Pixel_TCL cp = map.at(cur);
        circle(region, cur, (int)(cp.radius * config.radius_scaling), Scalar((unsigned char)255), -1);
        tclregion.at<uchar>(cur) = 255;

        int xmin = std::max(0, cur.x - search_distance);
        int xmax = std::min(width - 1, cur.x + search_distance);
        int ymin = std::max(0, cur.y - search_distance);
        int ymax = std::min(height - 1, cur.y + search_distance);

        for (int y = ymin; y <= ymax; y++)
        {
            for (int x = xmin; x <= xmax; x++)
            {
                Point2i curpt{x, y};
                if (search_mark.at<uchar>(curpt))
                    continue;
                Pixel_TCL np = map.at(curpt);
                if (np.tr > config.t_tr && np.tcl > config.t_tcl)
                {
                    // if (!(abs(np.radius - cp.radius) < config.t_rad * cp.radius))
                    // {
                    //     // printf("abs(np.radius - cp.radius) < config.t_rad * cp.radius\n");
                    //     continue;
                    // }
                    // if (!(abs(np.cos - cp.cos) < config.t_delta))
                    // {
                    //     // printf("abs(np.cos - cp.cos) < config.t_delta\n");
                    //     continue;
                    // }
                    // if (!(abs(np.sin - cp.sin) < config.t_delta || 2 - abs(np.sin) - abs(cp.sin) < config.t_delta))
                    // {
                    //     // printf("abs(np.sin - cp.sin) < config.t_delta\n");
                    //     continue;
                    // }
                    search_mark.at<uchar>(curpt) = 255;
                    // points.push_back(curpt);
                    to_search.push_back(curpt);
                    sum_radius += np.radius;
                    // sum_x += x;
                    // sum_y += y;
                    // sum_cos += np.cos;
                    // sum_sin += np.sin;
                    ptnumber++;
                }
            }
        }
    }

    // printf("%d\n", i);

    sum_radius /= ptnumber;
    // sum_x /= ptnumber;
    // sum_y /= ptnumber;
    // sum_sin /= ptnumber;
    // sum_cos /= ptnumber;

    int area_region = countNonZero(region);
    Mat andtr, andtcl;
    bitwise_and(region, trmap, andtr);
    bitwise_and(tclregion, trmap, andtcl);
    int area_text_region = countNonZero(andtr);
    int area_text_region_tcl = countNonZero(andtcl);

    // if (ptnumber < config.fewest_tcl)
    // {
    //     // printf("ptnumber < config.fewest_tcl\n");
    //     return;
    // }
    if ((float)ptnumber / sum_radius / sum_radius < config.fewest_tcl_ratio)
    {
        // printf("(float)ptnumber / sum_radius / sum_radius < config.fewest_tcl_ratio\n");
        return;
    }
    if ((float)area_text_region / area_region < config.smallest_area_ratio)
    {
        // printf("(float)area_text_region / area_region < config.smallest_area_ratio\n");
        return;
    }

    // if ((float)area_text_region_tcl / ptnumber < config.smallest_area_ratio_tcl)
    // {
    //     // printf("(float)area_text_region / area_region < config.smallest_area_ratio\n");
    //     return;
    // }

    regions.emplace_back();
    auto &r = regions.back();
    // r.avg_x = sum_x;
    // r.avg_y = sum_y;
    // r.avg_r = sum_radius;
    // r.avg_cos = sum_cos;
    // r.avg_sin = sum_sin;
    // r.region = region;
    // r.ptnumber = ptnumber;
    // r.area = area_text_region;
    vector<Vec4i> hierarchy;
    findContours(region, r.contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_L1);
    // printf("contours in single: %d\n", (int)r.contours.size());
}

bool PostProcessor::postprocess_tcl()
{
    InferenceMap<Pixel_TCL> &map = *(InferenceMap<Pixel_TCL> *)this->inferencemap;

    search_mark.create(height, width, CV_8UC1);
    search_mark.setTo(Scalar((unsigned char)0));

    trmap.create(height, width, CV_8UC1);

    // #pragma omp parallel for
    for (int i = 0; i < height; ++i)
    {
        uchar *tr = &trmap.at<uchar>(i, 0);
        const Pixel_TCL *pix = &map.at(0, i);
        for (int j = 0; j < width; ++j, ++pix, ++tr)
            *tr = pix->tr > config.t_tr ? 255 : 0;
    }

    vector<Point2i> ptlist = generate_random_vector(height, width);
    for (Point2i pt : ptlist)
    {
        if (map.at(pt).tcl > config.t_tcl && !search_mark.at<uchar>(pt))
        {
            search_contour(pt);
            // break;
        }
    }

    return true;
}

Point2i PostProcessor::topoint(int i)
{
    return Point2i(i % width, i / width);
}

int PostProcessor::toint(Point2i pt)
{
    return pt.x + pt.y * width;
}

bool PostProcessor::postprocess_pixellink()
{
    InferenceMap<Pixel_PixelLink> &map = *(InferenceMap<Pixel_PixelLink> *)this->inferencemap;

    disjointset dset;
    dset.init(width * height);

    int directions[][2] = {
        {-1, -1},
        {0, -1},
        {1, -1},
        {1, 0},
        {1, 1},
        {0, 1},
        {-1, 1},
        {-1, 0}};

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            Point2i curpt{j, i};
            if (map.at(curpt).tr < config.t_tr)
                continue;
            for (int k = 0; k < 8; ++k)
            {
                if (map.at(curpt).link[k] < config.t_tcl)
                    continue;
                Point2i newpt{curpt.x + directions[i][0], curpt.y + directions[i][1]};
                if (newpt.x < 0 || newpt.x >= width || newpt.y < 0 || newpt.y >= height)
                    continue;
                if (map.at(newpt).tr > config.t_tr)
                    dset.union_element(toint(curpt), toint(newpt));
            }
        }
    }

    std::map<int, Mat> contours;

    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            Point2i curpt{j, i};
            if(map.at(curpt).tr < config.t_tr)
                continue;
            int id = toint(curpt);
            id = dset.get_setid(id);
            auto it = contours.find(id);
            if(it == contours.end())
            {
                Mat mat;
                mat.create(height, width, CV_8UC1);
                mat.setTo(0);
                contours.insert(std::make_pair(id, mat));
            }
            it = contours.find(id);
            it->second.at<uchar>(curpt) = 255;
        }
    }

    vector<vector<Point> > newctn;
    vector<Vec4i> h_useless;

    for(auto& cnt: contours)
    {
        findContours(cnt.second, newctn, h_useless, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
        regions.emplace_back();
        auto &back = regions.back();
        back.contours = newctn;
    }

    return true;
}
