#include <arm_neon.h>
#include <stdio.h>
#include "inner.h"
#include "util.h"

#define LOGN 9

int main()
{
    const size_t FALCON_N = (size_t)1 << LOGN;
    fpr f_gold[FALCON_N], f_test[FALCON_N];
    fpr tmp;
    // int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp.v = i;
        f_gold[i] = tmp;
        f_test[i] = tmp;
    }
    // print_array(f_gold, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_NEON_FFT(f_gold, LOGN);
    // print_array(f_gold, FALCON_N, "FFT", 1);
    
    print_array(f_gold, FALCON_N, "Before iFFT", 1);
    PQCLEAN_FALCON512_NEON_iFFT(f_gold, LOGN);
    print_array(f_gold, FALCON_N, "iFFT", 1);

    return 0;
}