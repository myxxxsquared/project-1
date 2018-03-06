
#ifndef DISJOINTSET_HEADER
#define DISJOINTSET_HEADER

#include <vector>
using namespace std;

struct disjointset_element
{
    int parent;
    int rank;
};

class disjointset
{
  public:
    vector<disjointset_element> elements;

    void init(int n);
    void union_element(int i, int j);
    int get_setid(int i);
};

#endif