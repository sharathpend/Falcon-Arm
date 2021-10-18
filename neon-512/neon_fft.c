#include <arm_neon.h>
#include <stdio.h>
#include "inner.h"
#include "util.h"

int main()
{
    fpr f_gold[FALCON_N], f_test[FALCON_N];
    fpr f0_gold[FALCON_N], f1_gold[FALCON_N];
    fpr f0_test[FALCON_N], f1_test[FALCON_N];
    fpr tmp;
    int size = 8;
    int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp = i;
        f_gold[i] = tmp;
        f_test[i] = tmp;
        f0_gold[i] = tmp;
        f1_gold[i] = tmp;
    }
    // print_array(f_fft, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_NEON_iFFT(f_fft);
    // PQCLEAN_FALCON512_NEON_FFT(f_fft);
    // print_layer(f_fft, 16, FALCON_N);
    // print_array(f_fft, FALCON_N, "FFT", 1);
    
    PQCLEAN_FALCON512_NEON_poly_split_fft(f0_gold, f1_gold, f_gold, size);
    PQCLEAN_FALCON512_NEON_poly_merge_fft(f_test, f0_gold, f1_gold, size);

    print_array(f0_gold, 1 << (size - 1), "f0_gold", 1);
    print_array(f1_gold, 1 << (size - 1), "f1_gold", 1);
    print_array(f_test, 1 << size, "f_test", 1);

    ret |= compare(f_gold, f_test, 1 << size, "Compare with original");

    return 0;
}