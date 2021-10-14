#include "inner.h"

void PQCLEAN_FALCON512_NEON_mergeFFT_even(fpr *f, unsigned int logn)
{
    // Total: 32 = 16 + 8 + 8 register
    float64x2x4_t s_re_im, tmp;           // 8
    float64x2x4_t x_re, x_im, y_re, y_im; // 16
    float64x2x4_t v_re, v_im;             // 8
    float64x2x2_t s_tmp;                  // 2
    // Level 4, 5
    float64x2x2_t x_tmp, y_tmp;
    // Level 6, 7
    float64x2_t div_n;

    const unsigned int n = 1 << logn;
    const unsigned int hn = n >> 1;
    const unsigned int qn = n >> 2;

    // Level 0, 1, 2, 3
    
}