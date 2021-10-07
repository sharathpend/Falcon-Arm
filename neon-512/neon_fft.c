#include <arm_neon.h>
#include <stdio.h>
#include "inner.h"
#include "util.h"

#define FALCON_LOGN 9
#define FALCON_N (1 << FALCON_LOGN)

int main()
{
    fpr f_gold[FALCON_N], f_test[FALCON_N];
    fpr tmp;
    // int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp = i;
        f_gold[i] = tmp;
        f_test[i] = tmp;
    }
    // print_array(f_gold, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_NEON_FFT(f_gold, LOGN);
    // print_array(f_gold, FALCON_N, "FFT", 1);
    
    // print_array(f_gold, FALCON_N, "Before iFFT", 1);
    PQCLEAN_FALCON512_NEON_iFFT(f_gold, FALCON_LOGN);
    print_array(f_gold, FALCON_N, "iFFT", 1);

    return 0;
}