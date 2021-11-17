#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include "macrofx4.h"
/*
 * Convert an integer polynomial (with small values) into the
 * representation with complex numbers.
 * IMPORTANT: Correct, verified, optimized.
 */
void smallints_to_fpr1(double *r, const int8_t *t, const unsigned logn)
{
    float64x2x4_t neon_t0;
    int8x16x2_t neon_hm;
    int16x8_t neon_zero;
    int16x8x4_t neon_hmu16;
    int32x4x4_t neon_hmu32[2];
    int64x2x4_t neon_hms64[4];
    neon_zero = vdupq_n_s16(0);

    const unsigned falcon_n = 1 << logn;

    for (unsigned u = 0; u < falcon_n; u += 32)
    {
        neon_hm = vld1q_s8_x2(&t[u]);

        neon_hmu16.val[0] = (int16x8_t) vzip1q_s8(neon_hm.val[0], (int8x16_t) neon_zero);
        neon_hmu16.val[1] = (int16x8_t) vzip2q_s8(neon_hm.val[0], (int8x16_t) neon_zero);
        neon_hmu16.val[2] = (int16x8_t) vzip1q_s8(neon_hm.val[1], (int8x16_t) neon_zero);
        neon_hmu16.val[3] = (int16x8_t) vzip2q_s8(neon_hm.val[1], (int8x16_t) neon_zero);

        neon_hmu32[0].val[0] = (int32x4_t) vzip1q_s16(neon_hmu16.val[0], (int16x8_t) neon_zero);
        neon_hmu32[0].val[1] = (int32x4_t) vzip2q_s16(neon_hmu16.val[0], (int16x8_t) neon_zero);
        neon_hmu32[0].val[2] = (int32x4_t) vzip1q_s16(neon_hmu16.val[1], (int16x8_t) neon_zero);
        neon_hmu32[0].val[3] = (int32x4_t) vzip2q_s16(neon_hmu16.val[1], (int16x8_t) neon_zero);

        neon_hmu32[1].val[0] = (int32x4_t) vzip1q_s16(neon_hmu16.val[2], (int16x8_t) neon_zero);
        neon_hmu32[1].val[1] = (int32x4_t) vzip2q_s16(neon_hmu16.val[2], (int16x8_t) neon_zero);
        neon_hmu32[1].val[2] = (int32x4_t) vzip1q_s16(neon_hmu16.val[3], (int16x8_t) neon_zero);
        neon_hmu32[1].val[3] = (int32x4_t) vzip2q_s16(neon_hmu16.val[3], (int16x8_t) neon_zero);

        neon_hms64[0].val[0] = (int64x2_t) vzip1q_s32(neon_hmu32[0].val[0], (int32x4_t) neon_zero);
        neon_hms64[0].val[1] = (int64x2_t) vzip2q_s32(neon_hmu32[0].val[0], (int32x4_t) neon_zero);
        neon_hms64[0].val[2] = (int64x2_t) vzip1q_s32(neon_hmu32[0].val[1], (int32x4_t) neon_zero);
        neon_hms64[0].val[3] = (int64x2_t) vzip2q_s32(neon_hmu32[0].val[1], (int32x4_t) neon_zero);

        neon_hms64[1].val[0] = (int64x2_t) vzip1q_s32(neon_hmu32[0].val[2], (int32x4_t) neon_zero);
        neon_hms64[1].val[1] = (int64x2_t) vzip2q_s32(neon_hmu32[0].val[2], (int32x4_t) neon_zero);
        neon_hms64[1].val[2] = (int64x2_t) vzip1q_s32(neon_hmu32[0].val[3], (int32x4_t) neon_zero);
        neon_hms64[1].val[3] = (int64x2_t) vzip2q_s32(neon_hmu32[0].val[3], (int32x4_t) neon_zero);

        neon_hms64[2].val[0] = (int64x2_t) vzip1q_s32(neon_hmu32[1].val[0], (int32x4_t) neon_zero);
        neon_hms64[2].val[1] = (int64x2_t) vzip2q_s32(neon_hmu32[1].val[0], (int32x4_t) neon_zero);
        neon_hms64[2].val[2] = (int64x2_t) vzip1q_s32(neon_hmu32[1].val[1], (int32x4_t) neon_zero);
        neon_hms64[2].val[3] = (int64x2_t) vzip2q_s32(neon_hmu32[1].val[1], (int32x4_t) neon_zero);

        neon_hms64[3].val[0] = (int64x2_t) vzip1q_s32(neon_hmu32[1].val[2], (int32x4_t) neon_zero);
        neon_hms64[3].val[1] = (int64x2_t) vzip2q_s32(neon_hmu32[1].val[2], (int32x4_t) neon_zero);
        neon_hms64[3].val[2] = (int64x2_t) vzip1q_s32(neon_hmu32[1].val[3], (int32x4_t) neon_zero);
        neon_hms64[3].val[3] = (int64x2_t) vzip2q_s32(neon_hmu32[1].val[3], (int32x4_t) neon_zero);

        neon_t0.val[0] = vcvtq_f64_s64(neon_hms64[0].val[0]);
        neon_t0.val[1] = vcvtq_f64_s64(neon_hms64[0].val[1]);
        neon_t0.val[2] = vcvtq_f64_s64(neon_hms64[0].val[2]);
        neon_t0.val[3] = vcvtq_f64_s64(neon_hms64[0].val[3]);
        vstorex4(&r[u], neon_t0);

        neon_t0.val[0] = vcvtq_f64_s64(neon_hms64[1].val[0]);
        neon_t0.val[1] = vcvtq_f64_s64(neon_hms64[1].val[1]);
        neon_t0.val[2] = vcvtq_f64_s64(neon_hms64[1].val[2]);
        neon_t0.val[3] = vcvtq_f64_s64(neon_hms64[1].val[3]);
        vstorex4(&r[u + 8], neon_t0);

        neon_t0.val[0] = vcvtq_f64_s64(neon_hms64[2].val[0]);
        neon_t0.val[1] = vcvtq_f64_s64(neon_hms64[2].val[1]);
        neon_t0.val[2] = vcvtq_f64_s64(neon_hms64[2].val[2]);
        neon_t0.val[3] = vcvtq_f64_s64(neon_hms64[2].val[3]);
        vstorex4(&r[u + 16], neon_t0);

        neon_t0.val[0] = vcvtq_f64_s64(neon_hms64[3].val[0]);
        neon_t0.val[1] = vcvtq_f64_s64(neon_hms64[3].val[1]);
        neon_t0.val[2] = vcvtq_f64_s64(neon_hms64[3].val[2]);
        neon_t0.val[3] = vcvtq_f64_s64(neon_hms64[3].val[3]);
        vstorex4(&r[u + 24], neon_t0);
    }
}

/*
 * Convert an integer polynomial (with small values) into the
 * representation with complex numbers.
 * IMPORTANT: Correct, verified, optimized.
 */
void smallints_to_fpr(double *r, const int8_t *t, unsigned logn)
{
    const unsigned n = 1 << logn;
    float64x2x4_t neon_flo64, neon_fhi64;
    int64x2x4_t neon_lo64, neon_hi64;
    int32x4_t neon_lo32[2], neon_hi32[2];
    int16x8_t neon_lo16, neon_hi16;
    int8x16_t neon_8;

    for (unsigned i = 0; i < n; i += 16)
    {
        neon_8 = vld1q_s8(&t[i]);

        // Extend from 8 to 16 bit
        // x7 | x6 | x5 | x5 - x3 | x2 | x1 | x0
        neon_lo16 = vshll_n_s8(vget_low_s8(neon_8), 0);
        neon_hi16 = vshll_high_n_s8(neon_8, 0);

        // Extend from 16 to 32 bit
        // xxx3 | xxx2 | xxx1 | xxx0
        neon_lo32[0] = vshll_n_s16(vget_low_s16(neon_lo16), 0);
        neon_lo32[1] = vshll_high_n_s16(neon_lo16, 0);
        neon_hi32[0] = vshll_n_s16(vget_low_s16(neon_hi16), 0);
        neon_hi32[1] = vshll_high_n_s16(neon_hi16, 0);

        // Extend from 32 to 64 bit
        neon_lo64.val[0] = vshll_n_s32(vget_low_s32(neon_lo32[0]), 0);
        neon_lo64.val[1] = vshll_high_n_s32(neon_lo32[0], 0);
        neon_lo64.val[2] = vshll_n_s32(vget_low_s32(neon_lo32[1]), 0);
        neon_lo64.val[3] = vshll_high_n_s32(neon_lo32[1], 0);

        neon_hi64.val[0] = vshll_n_s32(vget_low_s32(neon_hi32[0]), 0);
        neon_hi64.val[1] = vshll_high_n_s32(neon_hi32[0], 0);
        neon_hi64.val[2] = vshll_n_s32(vget_low_s32(neon_hi32[1]), 0);
        neon_hi64.val[3] = vshll_high_n_s32(neon_hi32[1], 0);

        vfcvtx4(neon_flo64, neon_lo64);
        vfcvtx4(neon_fhi64, neon_hi64);

        vstorex4(&r[i], neon_flo64);
        vstorex4(&r[i + 8], neon_fhi64);
    }
}

#define NTEST 100
#define LOGN 6

int main()
{
    int8_t gold[1 << LOGN], test[1 << LOGN];
    int8_t tmp;
    double fgold[1 << LOGN], ftest[1 << LOGN];
    const unsigned falcon_n = 1 << LOGN;
    for (int i = 0; i < falcon_n; i++)
    {
        tmp = rand() & 0xff;
        gold[i] = tmp;
        test[i] = tmp;
    }
    smallints_to_fpr(fgold, gold, LOGN);
    smallints_to_fpr1(ftest, test, LOGN);

    for (int i = 0; i < falcon_n; i++)
    {
        if (ftest[i] != fgold[i])
        {
            printf("[%d]: %.20f != %.20f\n", i, fgold[i], ftest[i]);
            return 1;
        }
    }
    return 0;
}

/* 
 * Conclusion: int8_t sign bit is loss after zip
 */