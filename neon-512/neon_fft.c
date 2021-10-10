#include <arm_neon.h>
#include <stdio.h>
#include "inner.h"
#include "util.h"


int main()
{
    fpr f_ifft[FALCON_N], f_fft[FALCON_N];
    fpr tmp;
    // int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp = i;
        f_ifft[i] = tmp;
        f_fft[i] = tmp;
    }
    // print_array(f_fft, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_NEON_iFFT(f_fft);
    PQCLEAN_FALCON512_NEON_FFT(f_fft);
    // print_layer(f_fft, 16, FALCON_N);
    print_array(f_fft, FALCON_N, "FFT", 1);
    
    // print_array(f_ifft, FALCON_N, "Before iFFT", 1);
    // print_array(f_ifft, FALCON_N, "iFFT", 1);

    return 0;
}