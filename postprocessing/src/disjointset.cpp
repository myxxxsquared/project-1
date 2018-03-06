
#include "disjointset.hpp"

#include <cstdio>

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
    i = get_setid(i);
    j = get_setid(j);
    auto &iele = elements[i];
    auto &jele = elements[j];
    if (i == j)
        return;
    if (iele.rank < jele.rank)
        iele.parent = j;
    else if (iele.rank > jele.rank)
        jele.parent = i;
    else
    {
        iele.parent = j;
        jele.rank++;
    }
}

int disjointset::get_setid(int i)
{
    // if (elements[i].parent != i)
    //     elements[i].parent = get_setid(elements[i].parent);
    while (elements[i].parent != i)
        i = elements[i].parent;
    return i;
}