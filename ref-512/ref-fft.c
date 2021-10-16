#include "inner.h"
#include "util.h"
#include <stdio.h>

#define FALCON_LOGN 9
#define FALCON_N (1 << FALCON_LOGN)

int main()
{
    fpr f_gold[FALCON_N], f_test[FALCON_N];
    fpr f0_gold[FALCON_N], f1_gold[FALCON_N];
    fpr f0_test[FALCON_N], f1_test[FALCON_N];
    fpr tmp;
    int size = 3;
    int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp.v = i;
        f_gold[i] = tmp;
        f_test[i] = tmp;
    }
    // print_array(f_gold, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_CLEAN_iFFT(f_gold, FALCON_LOGN);
    // PQCLEAN_FALCON512_CLEAN_FFT(f_gold, FALCON_LOGN);
    // PQCLEAN_FALCON512_CLEAN_FFT_original(f_test, FALCON_LOGN);
    // print_layer(f_gold, 16, FALCON_N);
    // print_array(f_gold, FALCON_N, "FFT", 1);
    
    // print_array(f_gold, FALCON_N, "Before iFFT", 1);
    // PQCLEAN_FALCON512_CLEAN_iFFT_original(f_test, FALCON_LOGN);
    PQCLEAN_FALCON512_CLEAN_poly_split_fft(f0_gold, f1_gold, f_gold, size);
    // print_array(f_gold,  1 << (size  ), "f_gold", 1);
    // print_array(f0_gold, 1 << (size-1), "f0_gold", 1);
    // print_array(f1_gold, 1 << (size-1), "f1_gold", 1);
    PQCLEAN_FALCON512_CLEAN_poly_merge_fft(f_test, f0_gold, f1_gold, size);
    PQCLEAN_FALCON512_CLEAN_poly_merge_fft_original(f_gold, f0_gold, f1_gold, size);
    // print_array(f_test,  1 << (size  ), "f_test", 1);

    ret |= compare(f_gold, f_test, 1 << size, "Compare with original");

    return ret;
}