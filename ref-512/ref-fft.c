#include "inner.h"
#include "util.h"
#include <stdio.h>

#define LOGN 9

int main()
{
    const size_t FALCON_N = (size_t)1 << LOGN;
    fpr f_gold[FALCON_N], f_test[FALCON_N];
    fpr tmp;
    fpr tmp_fpr;
    // int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp.v = fRand(0, 12289);
        // tmp = 1;
        // printf("tmp = %f\n", tmp);
        // tmp_fpr = fpr_of(tmp);
        // printf("tmp_fpr = %f\n", tmp_fpr);
        // tmp = fpr_rint(tmp_fpr);
        // printf("tmp = %f\n", tmp);


        f_gold[i] = tmp;
        f_test[i] = tmp;
    }
    // // PQCLEAN_FALCON512_CLEAN_FFT(f_gold, LOGN);
    // // PQCLEAN_FALCON512_NEON_FFT(f_test,  LOGN);
    // // ret |= compare(f_gold, f_test, n, "FFT");

    // // for (int i = 0; i < n; i++)
    // // {
    // //     tmp = fRand(0, 12289);
    // //     f_gold[i] = (fpr) tmp;
    // //     f_test[i] = (fpr) tmp;
    // // }

    print_array(f_gold, FALCON_N, "Before iFFT", 1);
    PQCLEAN_FALCON512_CLEAN_iFFT(f_gold, LOGN);
    print_array(f_gold, FALCON_N, "iFFT", 1);

    // TODO: run this on cloud

    return 0;
}