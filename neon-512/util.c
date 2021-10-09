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
        tmp = a[i];
        if (print_float)
        {
            printf("%f, ", tmp);
        }
        else
        {
            printf("%lu, ", (uint64_t)tmp);
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
        if (a != b)
        {
            printf("Wrong [%d]: %lf != %f \n", i, a, b);
            return 1;
        }
    }
    printf("OK\n");
    return 0;
}

void print_vector(float64x2x4_t a)
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            printf("a[%d][%d] = %f\n", i, j, a.val[i][j]);
        }
    }
}

void print_layer(fpr *a, int length, int falcon_n)
{
    // for (int j = 0; j < 4; j++)
    // {
    //     for (int i = 0; i < length; i += 4)
    //     {
    //         printf("[%d] = %0.10f\n", i + j, a[i + j]);
    //     }
    // }

    // for (int j = 0; j < 4; j++)
    // {
    //     for (int i = 0; i < length; i += 4)
    //     {
    //         printf("[%d] = %0.10f\n", i + j + falcon_n / 2, a[i + j + falcon_n / 2]);
    //     }
    // }

    for (int i = 0; i < 16; i++)
    {
        printf("[%d] = %0.10f\n", i, a[i]);
    }
    for (int i = 0; i < 16; i++)
    {
        printf("[%d] = %0.10f\n", i + falcon_n / 2, a[i + falcon_n / 2]);
    }
}