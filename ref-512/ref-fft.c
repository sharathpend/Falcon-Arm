#include "inner.h"
#include "util.h"
#include <stdio.h>

#define FALCON_LOGN 9
#define FALCON_N (1 << FALCON_LOGN)

int main()
{
    fpr f_gold[FALCON_N], f_test[FALCON_N];
    fpr tmp;
    int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp.v = i;
        f_gold[i] = tmp;
        f_test[i] = tmp;
    }
    // print_array(f_gold, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_CLEAN_iFFT(f_gold, FALCON_LOGN);
    PQCLEAN_FALCON512_CLEAN_FFT(f_gold, FALCON_LOGN);
    // PQCLEAN_FALCON512_CLEAN_FFT_original(f_test, FALCON_LOGN);
    // print_layer(f_gold, 16, FALCON_N);
    print_array(f_gold, FALCON_N, "FFT", 1);
    
    // print_array(f_gold, FALCON_N, "Before iFFT", 1);
    // PQCLEAN_FALCON512_CLEAN_iFFT_original(f_test, FALCON_LOGN);
    // print_array(f_gold, FALCON_N, "iFFT", 1);
    // ret |= compare(f_gold, f_test, FALCON_N, "Compare with original");

    return ret;
}