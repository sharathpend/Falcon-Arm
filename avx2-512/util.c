#include "util.h"
#include "inner.h"
#include <stdio.h>
#include <stdlib.h>

void print_array(fpr *a, int length, const char *string, int print_float)
{
    printf("%s:\n", string);
    double tmp;
    for (int i = 0; i < length; i++)
    {
        // printf("[%d] = %d\n", i, a[i]);
        tmp = a[i].v;
        if (print_float)
        {
            printf("%f, ", tmp);
        }
        else
        {
            printf("%llu, ", (uint64_t) tmp);
        }
    }
    printf("\n==============\n");
}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    // printf("f = %f\n", f);
    return fMin + f * (fMax - fMin);
}

int compare(fpr *gold, fpr *test, int bound, const char *string)
{
    printf("%s: ", string);
    fpr a, b;
    for (int i = 0; i < bound; i++)
    {
        a = gold[i];
        b = test[i];
        if (a.v != b.v)
        {
            printf("Wrong [%d]: %lf != %f \n", i, a.v, b.v);
            return 1;
        }
    }
    printf("OK\n");
    return 0;
}
