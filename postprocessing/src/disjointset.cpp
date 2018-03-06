
#include "disjointset.hpp"

void disjointset::init(int n)
{
    elements.resize(n);
    for (int i = 0; i < n; ++i)
    {
        auto &element = elements[i];
        element.parent = i;
        element.rank = 0;
    }
}

void disjointset::union_element(int i, int j)
{
    int iroot = get_setid(i);
    int jroot = get_setid(j);
    auto &iele = elements[iroot];
    auto &jele = elements[jroot];
    if (iroot == jroot)
        return;
    if (iele.rank < jele.rank)
        iele.parent = jroot;
    else if (iele.rank > jele.rank)
        jele.parent = iroot;
    else
    {
        iele.parent = jroot;
        jele.rank++;
    }
}

int disjointset::get_setid(int i)
{
    if (elements[i].parent != i)
        elements[i].parent = get_setid(elements[i].parent);
    return elements[i].parent;
}