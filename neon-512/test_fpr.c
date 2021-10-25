#include "fpr.h"
#include "vfpr.h"
#include "macro.h"
#include "util.h"
#include "inner.h"
#include <stdio.h>

#define TESTS 1000000

int main()
{
    fpr tmp, ccs;
    fpr x;
    float64x2_t neon_x, neon_ccs;
    uint64x2_t neon_ret;
    uint64_t ret;
    
    srand(123);

    for (int i = 0; i < TESTS; i++)
    {
        ccs = fRand(0, 0.6931471805599453);
        x = fRand(-10, 10);

        neon_x = vdupq_n_f64(x);
        neon_ccs = vdupq_n_f64(ccs);

        neon_ret = vfpr_expm_p63(neon_x, neon_ccs);
        ret = fpr_expm_p63(x, ccs);

        if ((ret != neon_ret[0]) || (ret != neon_ret[1]))
        {
            printf("Wrong vfpr_expm_p63 %d: %f -- %f\n", i, x, ccs);
            printf("Wrong vfpr_expm_p63 %d: %lx != %lx \n", i, ret, neon_ret[0]);
            return 1;
        }

        x = fRand(-FALCON_N, FALCON_N);
        neon_x = vdupq_n_f64(x);

        ret = (uint64_t) fpr_rint(x);
        neon_ret = (uint64x2_t) vfpr_rint(  neon_x);
        // neon_ret = (uint64x2_t) vcvtnq_s64_f64(neon_x);

        if ((ret != neon_ret[0]) || (ret != neon_ret[1]))
        {
            printf("Wrong vfpr_rint %d: %.20f \n", i, x);
            printf("Wrong vfpr_rint %d: %lx != %lx \n", i, ret, neon_ret[0]);
            return 1;
        }
        
        ret = (uint64_t) fpr_floor(x);
        neon_ret = (uint64x2_t) vfpr_floor(neon_x);

        if ((ret != neon_ret[0]) || (ret != neon_ret[1]))
        {
            printf("Wrong vfpr_floor %d: %.20f \n", i, x);
            printf("Wrong vfpr_floor %d: %d != %d \n", i, ret, neon_ret[0]);
            return 1;
        }
        else{
            // printf("Wrong vfpr_floor %d: %.20f \n", i, x);
            // printf("Wrong vfpr_floor %d: %lx != %lx \n", i, ret, neon_ret[0]);
            // return 1;
        }

    }
    printf("OK\n");
    return 0;
}