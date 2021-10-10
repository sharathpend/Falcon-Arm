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
            // printf("[%d] = %f\n", i, tmp);
        }
        else
        {
            printf("%llu, ", (uint64_t) tmp);
        }
    }
    printf("\n==============\n");
}
void print_layer(fpr *a, int length, int falcon_n)
{
    // for (int j = 0; j < 4; j++)
    // {
    //     for (int i = 0; i < length; i+=4)
    //     {
    //         printf("[%d] = %0.10f\n", i + j, a[i + j]);
    //     }
    // }

    // for (int j = 0; j < 4; j++)
    // {
    //     for (int i = 0; i < length; i+=4)
    //     {
    //         printf("[%d] = %0.10f\n", i + j + falcon_n/2, a[i + j + falcon_n/2]);
    //     }
    // }
    for (int i = 0; i < 16; i++)
    {
        printf("[%d] = %0.10f\n", i, a[i]);
    }
    for (int i = 0; i < 16; i++)
    {
        printf("[%d] = %0.10f\n", i + falcon_n/2, a[i + falcon_n/2]);
    }
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
