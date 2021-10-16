#include <arm_neon.h>
#include <stdio.h>
#include "inner.h"
#include "util.h"

int main()
{
    fpr f_ifft[FALCON_N], f_fft[FALCON_N];
    fpr f0_gold[FALCON_N], f1_gold[FALCON_N];
    fpr f0_test[FALCON_N], f1_test[FALCON_N];
    fpr f_test[FALCON_N];
    fpr tmp;
    int size = 7;
    int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp = i;
        f_ifft[i] = tmp;
        f_fft[i] = tmp;
        f0_gold[i] = tmp;
        f1_gold[i] = tmp;
    }
    // print_array(f_fft, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_NEON_iFFT(f_fft);
    // PQCLEAN_FALCON512_NEON_FFT(f_fft);
    // print_layer(f_fft, 16, FALCON_N);
    // print_array(f_fft, FALCON_N, "FFT", 1);

    PQCLEAN_FALCON512_NEON_poly_merge_fft(f_test, f0_gold, f1_gold, size);
    print_array(f_test, 1 << size, "f_test", 1);

    // print_array(f_ifft, FALCON_N, "Before iFFT", 1);

    return 0;
}