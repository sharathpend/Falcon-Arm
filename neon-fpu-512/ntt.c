#include <arm_neon.h>
#include "macrous.h"
#include "inner.h"
#include "ntt.h"
#include "ntt_consts.h"
#include "config.h"

/*
 * Assume Input in the range [-Q/2, Q/2]
 * Total Barrett point for N = 512, 1024: 2048, 4096
 */
void neon_fwdNTT(int16_t a[FALCON_N], const char mont)
{
    // Total SIMD registers 29 = 16 + 12 + 1
    int16x8x4_t v0, v1, v2, v3; // 16
    int16x8x4_t zl, zh, t;      // 12
    int16x8x2_t zlh, zhh;       // 4
    int16x8_t neon_qmvq;        // 1
    unsigned k = 0;

    neon_qmvq = vld1q_s16(qmvq);
    zl.val[0] = vld1q_s16(&ntt_br[k]);
    zh.val[0] = vld1q_s16(&ntt_qinv_br[k]);
    k += 8;
#if FALCON_N == 512
    // Layer 8, 7
    for (unsigned j = 0; j < 128; j += 32)
    {
        vload_s16_x4(v0, &a[j]);
        vload_s16_x4(v1, &a[j + 128]);
        vload_s16_x4(v2, &a[j + 256]);
        vload_s16_x4(v3, &a[j + 384]);

        // v0: .5
        // v1: .5
        // v2: .5
        // v3: .5

        // Layer 8
        // v0 - v2, v1 - v3
        ctbf_bri_x4(v0, v2, zl.val[0], zh.val[0], 1, 1, 1, 1, neon_qmvq, t);
        ctbf_bri_x4(v1, v3, zl.val[0], zh.val[0], 1, 1, 1, 1, neon_qmvq, t);

        // v0: 1.2
        // v1: 1.2
        // v2: 1.2
        // v3: 1.2

        // Layer 7
        // v0 - v1, v2 - v3
        ctbf_bri_x4(v0, v1, zl.val[0], zh.val[0], 2, 2, 2, 2, neon_qmvq, t);
        ctbf_bri_x4(v2, v3, zl.val[0], zh.val[0], 3, 3, 3, 3, neon_qmvq, t);

        // 2.14 -> 0.5
        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);

        // Store at 0.5Q
        vstore_s16_x4(&a[j], v0);
        vstore_s16_x4(&a[j + 128], v1);
        vstore_s16_x4(&a[j + 256], v2);
        vstore_s16_x4(&a[j + 384], v3);
    }
#elif FALCON_N == 1024
    // Layer 9, 8, 7
    int16x8x2_t u0, u1, u2, u3, u4, u5, u6, u7;

    for (unsigned j = 0; j < 128; j += 16)
    {
        vload_s16_x2(u0, &a[j]);
        vload_s16_x2(u1, &a[j + 128]);
        vload_s16_x2(u2, &a[j + 256]);
        vload_s16_x2(u3, &a[j + 384]);

        vload_s16_x2(u4, &a[j + 512]);
        vload_s16_x2(u5, &a[j + 640]);
        vload_s16_x2(u6, &a[j + 768]);
        vload_s16_x2(u7, &a[j + 896]);

        // u0, 4: .5
        // u1, 5: .5
        // u2, 6: .5
        // u3, 7: .5

        // Layer 9
        // u0 - u4, u1 - u5
        // u2 - u6, u3 - u7
        ctbf_bri(u0.val[0], u4.val[0], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[0]);
        ctbf_bri(u0.val[1], u4.val[1], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[1]);
        ctbf_bri(u1.val[0], u5.val[0], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[2]);
        ctbf_bri(u1.val[1], u5.val[1], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[3]);

        ctbf_bri(u2.val[0], u6.val[0], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[0]);
        ctbf_bri(u2.val[1], u6.val[1], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[1]);
        ctbf_bri(u3.val[0], u7.val[0], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[2]);
        ctbf_bri(u3.val[1], u7.val[1], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[3]);

        // u0, 4: 1.2
        // u1, 5: 1.2
        // u2, 6: 1.2
        // u3, 7: 1.2

        // Layer 8
        // u0 - u2, u1 - u3
        // u4 - u6, u5 - u7
        ctbf_bri(u0.val[0], u2.val[0], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[0]);
        ctbf_bri(u0.val[1], u2.val[1], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[1]);
        ctbf_bri(u1.val[0], u3.val[0], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[2]);
        ctbf_bri(u1.val[1], u3.val[1], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[3]);

        ctbf_bri(u4.val[0], u6.val[0], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[0]);
        ctbf_bri(u4.val[1], u6.val[1], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[1]);
        ctbf_bri(u5.val[0], u7.val[0], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[2]);
        ctbf_bri(u5.val[1], u7.val[1], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[3]);

        // 2.14 -> 0.5
        barrett_x2(u0, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u1, 0, 1, 2, 3, neon_qmvq, t);
        barrett_x2(u2, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u3, 0, 1, 2, 3, neon_qmvq, t);

        barrett_x2(u4, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u5, 0, 1, 2, 3, neon_qmvq, t);
        barrett_x2(u6, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u7, 0, 1, 2, 3, neon_qmvq, t);
        // u0, 4: .5
        // u1, 5: .5
        // u2, 6: .5
        // u3, 7: .5

        // Layer 7
        // u0 - u1, u2 - u3
        // u4 - u5, u6 - u7
        ctbf_bri(u0.val[0], u1.val[0], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[0]);
        ctbf_bri(u0.val[1], u1.val[1], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[1]);
        ctbf_bri(u2.val[0], u3.val[0], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[2]);
        ctbf_bri(u2.val[1], u3.val[1], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[3]);

        ctbf_bri(u4.val[0], u5.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[0]);
        ctbf_bri(u4.val[1], u5.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[1]);
        ctbf_bri(u6.val[0], u7.val[0], zl.val[0], zh.val[0], 7, neon_qmvq, t.val[2]);
        ctbf_bri(u6.val[1], u7.val[1], zl.val[0], zh.val[0], 7, neon_qmvq, t.val[3]);

        // u0, 4: 1.2
        // u1, 5: 1.2
        // u2, 6: 1.2
        // u3, 7: 1.2

        // Store at 1.2Q
        vstore_s16_x2(&a[j], u0);
        vstore_s16_x2(&a[j + 128], u1);
        vstore_s16_x2(&a[j + 256], u2);
        vstore_s16_x2(&a[j + 384], u3);

        vstore_s16_x2(&a[j + 512], u4);
        vstore_s16_x2(&a[j + 640], u5);
        vstore_s16_x2(&a[j + 768], u6);
        vstore_s16_x2(&a[j + 896], u7);
    }
#else
#error "FALCON_N is either 512 or 1024"
#endif

    // Layer 6, 5, 4, 3, 2, 1, 0
    for (unsigned j = 0; j < FALCON_N; j += 128)
    {
        vload_s16_x4(v0, &a[j]);
        vload_s16_x4(v1, &a[j + 32]);
        vload_s16_x4(v2, &a[j + 64]);
        vload_s16_x4(v3, &a[j + 96]);

        vload_s16_x2(zlh, &ntt_br[k]);
        vload_s16_x2(zhh, &ntt_qinv_br[k]);
        k += 16;

        // Layer 6
        // v0 - v2, v1 - v3
        ctbf_bri_x4(v0, v2, zlh.val[0], zhh.val[0], 0, 0, 0, 0, neon_qmvq, t);
        ctbf_bri_x4(v1, v3, zlh.val[0], zhh.val[0], 0, 0, 0, 0, neon_qmvq, t);

#if FALCON_N == 1024
        // 2.3 -> 0.5
        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);
#elif FALCON_N == 512
        // 1.3
#endif

        // Layer 5
        // v0 - v1, v2 - v3
        ctbf_bri_x4(v0, v1, zlh.val[0], zhh.val[0], 1, 1, 1, 1, neon_qmvq, t);
        ctbf_bri_x4(v2, v3, zlh.val[0], zhh.val[0], 2, 2, 2, 2, neon_qmvq, t);

#if FALCON_N == 1024
        // 1.3
#elif FALCON_N == 512
        // 2.3 -> 0.5
        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);
#endif

        // Layer 4
        // v0(0, 1 - 2, 3)
        // v1(0, 1 - 2, 3)
        // v2(0, 1 - 2, 3)
        // v3(0, 1 - 2, 3)
        ctbf_bri(v0.val[0], v0.val[2], zlh.val[0], zhh.val[0], 3, neon_qmvq, t.val[0]);
        ctbf_bri(v0.val[1], v0.val[3], zlh.val[0], zhh.val[0], 3, neon_qmvq, t.val[1]);
        ctbf_bri(v1.val[0], v1.val[2], zlh.val[0], zhh.val[0], 4, neon_qmvq, t.val[2]);
        ctbf_bri(v1.val[1], v1.val[3], zlh.val[0], zhh.val[0], 4, neon_qmvq, t.val[3]);

        ctbf_bri(v2.val[0], v2.val[2], zlh.val[0], zhh.val[0], 5, neon_qmvq, t.val[0]);
        ctbf_bri(v2.val[1], v2.val[3], zlh.val[0], zhh.val[0], 5, neon_qmvq, t.val[1]);
        ctbf_bri(v3.val[0], v3.val[2], zlh.val[0], zhh.val[0], 6, neon_qmvq, t.val[2]);
        ctbf_bri(v3.val[1], v3.val[3], zlh.val[0], zhh.val[0], 6, neon_qmvq, t.val[3]);

#if FALCON_N == 1024
        // 2.3 -> 0.5
        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);
#elif FALCON_N == 512
        // 1.3
#endif

        // Layer 3
        // v0(0, 2 - 1, 3)
        // v1(0, 2 - 1, 3)
        // v2(0, 2 - 1, 3)
        // v3(0, 2 - 1, 3)
        ctbf_bri(v0.val[0], v0.val[1], zlh.val[0], zhh.val[0], 7, neon_qmvq, t.val[0]);
        ctbf_bri(v0.val[2], v0.val[3], zlh.val[1], zhh.val[1], 0, neon_qmvq, t.val[1]);
        ctbf_bri(v1.val[0], v1.val[1], zlh.val[1], zhh.val[1], 1, neon_qmvq, t.val[2]);
        ctbf_bri(v1.val[2], v1.val[3], zlh.val[1], zhh.val[1], 2, neon_qmvq, t.val[3]);

        ctbf_bri(v2.val[0], v2.val[1], zlh.val[1], zhh.val[1], 3, neon_qmvq, t.val[0]);
        ctbf_bri(v2.val[2], v2.val[3], zlh.val[1], zhh.val[1], 4, neon_qmvq, t.val[1]);
        ctbf_bri(v3.val[0], v3.val[1], zlh.val[1], zhh.val[1], 5, neon_qmvq, t.val[2]);
        ctbf_bri(v3.val[2], v3.val[3], zlh.val[1], zhh.val[1], 6, neon_qmvq, t.val[3]);

#if FALCON_N == 1024
        // 1.3
#elif FALCON_N == 512
        // 2.3 -> 0.5
        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);
#endif

        // Layer 2
        // Input:
        // 0,  1,  2,  3  | 4,  5,  6,  7
        // 8,  9,  10, 11 | 12, 13, 14, 15
        // 16, 17, 18, 19 | 20, 21, 22, 23
        // 24, 25, 26, 27 | 28, 29, 30, 31
        arrange(t, v0, 0, 2, 1, 3, 0, 1, 2, 3);
        v0 = t;
        arrange(t, v1, 0, 2, 1, 3, 0, 1, 2, 3);
        v1 = t;
        arrange(t, v2, 0, 2, 1, 3, 0, 1, 2, 3);
        v2 = t;
        arrange(t, v3, 0, 2, 1, 3, 0, 1, 2, 3);
        v3 = t;
        // Output:
        // 0,  1,  2,  3  | 16, 17, 18, 19
        // 4,  5,  6,  7  | 20, 21, 22, 23
        // 8,  9,  10, 11 | 24, 25, 26, 27
        // 12, 13, 14, 15 | 28, 29, 30, 31
        vload_s16_x4(zl, &ntt_br[k]);
        vload_s16_x4(zh, &ntt_qinv_br[k]);
        k += 32;

        ctbf_br(v0.val[0], v0.val[1], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        ctbf_br(v1.val[0], v1.val[1], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        ctbf_br(v2.val[0], v2.val[1], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        ctbf_br(v3.val[0], v3.val[1], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

        vload_s16_x4(zl, &ntt_br[k]);
        vload_s16_x4(zh, &ntt_qinv_br[k]);
        k += 32;

        ctbf_br(v0.val[2], v0.val[3], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        ctbf_br(v1.val[2], v1.val[3], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        ctbf_br(v2.val[2], v2.val[3], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        ctbf_br(v3.val[2], v3.val[3], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

#if FALCON_N == 1024
        // 2.3 -> 0.5
        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);
#elif FALCON_N == 512
        // 1.3
#endif

        // Layer 1: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
        // v0.val[0]: 0,  1,  2,  3  | 16, 17, 18, 19
        // v0.val[1]: 4,  5,  6,  7  | 20, 21, 22, 23
        // v0.val[2]: 8,  9,  10, 11 | 24, 25, 26, 27
        // v0.val[3]: 12, 13, 14, 15 | 28, 29, 30, 31
        // transpose 4x4
        transpose(v0, t);
        transpose(v1, t);
        transpose(v2, t);
        transpose(v3, t);
        // v0.val[0]: 0, 4, 8,  12 | 16, 20, 24, 28
        // v0.val[1]: 1, 5, 9,  13 | 17, 21, 25, 29
        // v0.val[2]: 2, 6, 10, 14 | 18, 22, 26, 30
        // v0.val[3]: 3, 7, 11, 15 | 19, 23, 27, 31

        vload_s16_x4(zl, &ntt_br[k]);
        vload_s16_x4(zh, &ntt_qinv_br[k]);
        k += 32;

        ctbf_br(v0.val[0], v0.val[2], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        ctbf_br(v0.val[1], v0.val[3], zl.val[0], zh.val[0], neon_qmvq, t.val[1]);
        ctbf_br(v1.val[0], v1.val[2], zl.val[1], zh.val[1], neon_qmvq, t.val[2]);
        ctbf_br(v1.val[1], v1.val[3], zl.val[1], zh.val[1], neon_qmvq, t.val[3]);

        ctbf_br(v2.val[0], v2.val[2], zl.val[2], zh.val[2], neon_qmvq, t.val[0]);
        ctbf_br(v2.val[1], v2.val[3], zl.val[2], zh.val[2], neon_qmvq, t.val[1]);
        ctbf_br(v3.val[0], v3.val[2], zl.val[3], zh.val[3], neon_qmvq, t.val[2]);
        ctbf_br(v3.val[1], v3.val[3], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

#if FALCON_N == 1024
        // 1.3
#elif FALCON_N == 512
        // 2.3 -> 0.5
        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);
#endif

        // Layer 0
        // v(0, 2 - 1, 3)
        vload_s16_x4(zl, &ntt_br[k]);
        vload_s16_x4(zh, &ntt_qinv_br[k]);
        k += 32;

        ctbf_br(v0.val[0], v0.val[1], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        ctbf_br(v1.val[0], v1.val[1], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        ctbf_br(v2.val[0], v2.val[1], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        ctbf_br(v3.val[0], v3.val[1], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

        vload_s16_x4(zl, &ntt_br[k]);
        vload_s16_x4(zh, &ntt_qinv_br[k]);
        k += 32;

        ctbf_br(v0.val[2], v0.val[3], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        ctbf_br(v1.val[2], v1.val[3], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        ctbf_br(v2.val[2], v2.val[3], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        ctbf_br(v3.val[2], v3.val[3], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

#if FALCON_N == 1024
        // 2.3
#elif FALCON_N == 512
        // 1.3
#endif

        if (mont)
        {
            // Convert to Montgomery domain by multiply with FALCON_MONT
            barmuli_mont_x4(v0, neon_qmvq, t);
            barmuli_mont_x4(v1, neon_qmvq, t);
            barmuli_mont_x4(v2, neon_qmvq, t);
            barmuli_mont_x4(v3, neon_qmvq, t);
        }

        vstore_s16_4(&a[j], v0);
        vstore_s16_4(&a[j + 32], v1);
        vstore_s16_4(&a[j + 64], v2);
        vstore_s16_4(&a[j + 96], v3);
    }
}

/*
 * Assume input in range [-Q, Q]
 * Total Barrett point N = 512, 1024: 1792, 3840
 */
void neon_invNTT(int16_t a[FALCON_N])
{
    // Total SIMD registers: 29 = 16 + 12 + 1
    int16x8x4_t v0, v1, v2, v3; // 16
    int16x8x4_t zl, zh, t;      // 12
    int16x8x2_t zlh, zhh;       // 4
    int16x8_t neon_qmvq;        // 1

    neon_qmvq = vld1q_s16(qmvq);
    unsigned j, k = 0;

    // Layer 0, 1, 2, 3, 4, 5, 6
    for (j = 0; j < FALCON_N; j += 128)
    {
        vload_s16_4(v0, &a[j]);
        vload_s16_4(v1, &a[j + 32]);
        vload_s16_4(v2, &a[j + 64]);
        vload_s16_4(v3, &a[j + 96]);

        // Layer 0
        // v0.val[0]: 0, 4, 8,  12 | 16, 20, 24, 28
        // v0.val[1]: 1, 5, 9,  13 | 17, 21, 25, 29
        // v0.val[2]: 2, 6, 10, 14 | 18, 22, 26, 30
        // v0.val[3]: 3, 7, 11, 15 | 19, 23, 27, 31
        vload_s16_x4(zl, &invntt_br[k]);
        vload_s16_x4(zh, &invntt_qinv_br[k]);
        k += 32;

        // 0 - 1*, 2 - 3*
        gsbf_br(v0.val[0], v0.val[1], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        gsbf_br(v1.val[0], v1.val[1], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        gsbf_br(v2.val[0], v2.val[1], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        gsbf_br(v3.val[0], v3.val[1], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

        vload_s16_x4(zl, &invntt_br[k]);
        vload_s16_x4(zh, &invntt_qinv_br[k]);
        k += 32;

        gsbf_br(v0.val[2], v0.val[3], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        gsbf_br(v1.val[2], v1.val[3], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        gsbf_br(v2.val[2], v2.val[3], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        gsbf_br(v3.val[2], v3.val[3], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

        // 0: 2
        // 1: 1.3
        // 2: 2
        // 3: 1.3

        barrett(v0.val[0], neon_qmvq, t.val[0]);
        barrett(v1.val[0], neon_qmvq, t.val[1]);
        barrett(v2.val[0], neon_qmvq, t.val[2]);
        barrett(v3.val[0], neon_qmvq, t.val[3]);

        // 0: 0.5
        // 1: 1.3
        // 2: 2
        // 3: 1.3

        // Layer 1
        // v0.val[0]: 0, 4, 8,  12 | 16, 20, 24, 28
        // v0.val[1]: 1, 5, 9,  13 | 17, 21, 25, 29
        // v0.val[2]: 2, 6, 10, 14 | 18, 22, 26, 30
        // v0.val[3]: 3, 7, 11, 15 | 19, 23, 27, 31
        // 0 - 2*, 1 - 3*

        vload_s16_x4(zl, &invntt_br[k]);
        vload_s16_x4(zh, &invntt_qinv_br[k]);
        k += 32;

        gsbf_br(v0.val[0], v0.val[2], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        gsbf_br(v0.val[1], v0.val[3], zl.val[0], zh.val[0], neon_qmvq, t.val[1]);
        gsbf_br(v1.val[0], v1.val[2], zl.val[1], zh.val[1], neon_qmvq, t.val[2]);
        gsbf_br(v1.val[1], v1.val[3], zl.val[1], zh.val[1], neon_qmvq, t.val[3]);

        gsbf_br(v2.val[0], v2.val[2], zl.val[2], zh.val[2], neon_qmvq, t.val[0]);
        gsbf_br(v2.val[1], v2.val[3], zl.val[2], zh.val[2], neon_qmvq, t.val[1]);
        gsbf_br(v3.val[0], v3.val[2], zl.val[3], zh.val[3], neon_qmvq, t.val[2]);
        gsbf_br(v3.val[1], v3.val[3], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

        // 0: 2.5
        // 1: 2.6
        // 2: 1.5
        // 3: 1.5

        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);

        // 0: 0.5
        // 1: 0.5
        // 2: 0.5
        // 3: 0.5

        // Layer 2
        // Before Transpose
        // v0.val[0]: 0, 4, 8,  12 | 16, 20, 24, 28
        // v0.val[1]: 1, 5, 9,  13 | 17, 21, 25, 29
        // v0.val[2]: 2, 6, 10, 14 | 18, 22, 26, 30
        // v0.val[3]: 3, 7, 11, 15 | 19, 23, 27, 31
        transpose(v0, t);
        transpose(v1, t);
        transpose(v2, t);
        transpose(v3, t);

        // After Transpose
        // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
        // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
        // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
        // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31
        // 0 - 1*, 2 - 3*
        vload_s16_x4(zl, &invntt_br[k]);
        vload_s16_x4(zh, &invntt_qinv_br[k]);
        k += 32;

        gsbf_br(v0.val[0], v0.val[1], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        gsbf_br(v1.val[0], v1.val[1], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        gsbf_br(v2.val[0], v2.val[1], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        gsbf_br(v3.val[0], v3.val[1], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

        vload_s16_x4(zl, &invntt_br[k]);
        vload_s16_x4(zh, &invntt_qinv_br[k]);
        k += 32;

        gsbf_br(v0.val[2], v0.val[3], zl.val[0], zh.val[0], neon_qmvq, t.val[0]);
        gsbf_br(v1.val[2], v1.val[3], zl.val[1], zh.val[1], neon_qmvq, t.val[1]);
        gsbf_br(v2.val[2], v2.val[3], zl.val[2], zh.val[2], neon_qmvq, t.val[2]);
        gsbf_br(v3.val[2], v3.val[3], zl.val[3], zh.val[3], neon_qmvq, t.val[3]);

        // 0: 1
        // 1: 0.9
        // 2: 1
        // 3: 0.9

        // Layer 3
        // Re-arrange vector from
        // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
        // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
        // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
        // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31
        // Compiler will handle register re-naming
        arrange(t, v0, 0, 1, 2, 3, 0, 2, 1, 3);
        v0 = t;

        // Compiler will handle register re-naming
        arrange(t, v1, 0, 1, 2, 3, 0, 2, 1, 3);
        v1 = t;

        // Compiler will handle register re-naming
        arrange(t, v2, 0, 1, 2, 3, 0, 2, 1, 3);
        v2 = t;

        // Compiler will handle register re-naming
        arrange(t, v3, 0, 1, 2, 3, 0, 2, 1, 3);
        v3 = t;
        // To
        // v0.val[0]: 0,  1,  2,  3  | 4,  5,  6,  7
        // v0.val[1]: 8,  9,  10, 11 | 12, 13, 14, 15
        // v0.val[2]: 16, 17, 18, 19 | 20, 21, 22, 23
        // v0.val[3]: 24, 25, 26, 27 | 28, 29, 30, 31
        // 0 - 1, 2 - 3
        vload_s16_x2(zlh, &invntt_br[k]);
        vload_s16_x2(zhh, &invntt_qinv_br[k]);
        k += 16;

        gsbf_bri(v0.val[0], v0.val[1], zlh.val[0], zhh.val[0], 0, neon_qmvq, t.val[0]);
        gsbf_bri(v0.val[2], v0.val[3], zlh.val[0], zhh.val[0], 1, neon_qmvq, t.val[1]);
        gsbf_bri(v1.val[0], v1.val[1], zlh.val[0], zhh.val[0], 2, neon_qmvq, t.val[2]);
        gsbf_bri(v1.val[2], v1.val[3], zlh.val[0], zhh.val[0], 3, neon_qmvq, t.val[3]);

        gsbf_bri(v2.val[0], v2.val[1], zlh.val[0], zhh.val[0], 4, neon_qmvq, t.val[0]);
        gsbf_bri(v2.val[2], v2.val[3], zlh.val[0], zhh.val[0], 5, neon_qmvq, t.val[1]);
        gsbf_bri(v3.val[0], v3.val[1], zlh.val[0], zhh.val[0], 6, neon_qmvq, t.val[2]);
        gsbf_bri(v3.val[2], v3.val[3], zlh.val[0], zhh.val[0], 7, neon_qmvq, t.val[3]);

        // 0: 2
        // 1: 1.3
        // 2: 2
        // 3: 1.3

        barrett(v0.val[0], neon_qmvq, t.val[0]);
        barrett(v1.val[0], neon_qmvq, t.val[1]);
        barrett(v2.val[0], neon_qmvq, t.val[2]);
        barrett(v3.val[0], neon_qmvq, t.val[3]);

        // 0: 0.5
        // 1: 1.3
        // 2: 2
        // 3: 1.3

        // Layer 4
        // v0.val[0]: 0,  1,  2,  3  | 4,  5,  6,  7
        // v0.val[1]: 8,  9,  10, 11 | 12, 13, 14, 15
        // v0.val[2]: 16, 17, 18, 19 | 20, 21, 22, 23
        // v0.val[3]: 24, 25, 26, 27 | 28, 29, 30, 31
        // 0 - 2, 1 - 3

        gsbf_bri(v0.val[0], v0.val[2], zlh.val[1], zhh.val[1], 0, neon_qmvq, t.val[0]);
        gsbf_bri(v0.val[1], v0.val[3], zlh.val[1], zhh.val[1], 0, neon_qmvq, t.val[1]);
        gsbf_bri(v1.val[0], v1.val[2], zlh.val[1], zhh.val[1], 1, neon_qmvq, t.val[2]);
        gsbf_bri(v1.val[1], v1.val[3], zlh.val[1], zhh.val[1], 1, neon_qmvq, t.val[3]);

        gsbf_bri(v2.val[0], v2.val[2], zlh.val[1], zhh.val[1], 2, neon_qmvq, t.val[0]);
        gsbf_bri(v2.val[1], v2.val[3], zlh.val[1], zhh.val[1], 2, neon_qmvq, t.val[1]);
        gsbf_bri(v3.val[0], v3.val[2], zlh.val[1], zhh.val[1], 3, neon_qmvq, t.val[2]);
        gsbf_bri(v3.val[1], v3.val[3], zlh.val[1], zhh.val[1], 3, neon_qmvq, t.val[3]);

        // 0: 2.5
        // 1: 2.5
        // 2: 1.5
        // 3: 1.5

        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);

        // 0: 0.5
        // 1: 0.5
        // 2: 0.5
        // 3: 0.5

        // Layer 5
        // Cross block
        // v0.0->3 - v1.0->3
        gsbf_bri_x4(v0, v1, zlh.val[1], zhh.val[1], 4, 4, 4, 4, neon_qmvq, t);
        gsbf_bri_x4(v2, v3, zlh.val[1], zhh.val[1], 5, 5, 5, 5, neon_qmvq, t);

        // v0: 1
        // v1: 0.9
        // v2: 1
        // v3: 0.9

        // Layer 6
        // Cross block
        // v0.0->3 - v2.0->3
        gsbf_bri_x4(v0, v2, zlh.val[1], zhh.val[1], 6, 6, 6, 6, neon_qmvq, t);
        gsbf_bri_x4(v1, v3, zlh.val[1], zhh.val[1], 6, 6, 6, 6, neon_qmvq, t);

        // v0: 2
        // v1: 1.8
        // v2: 1.3
        // v3: 1.2

        vstore_s16_x4(&a[j], v0);
        vstore_s16_x4(&a[j + 32], v1);
        vstore_s16_x4(&a[j + 64], v2);
        vstore_s16_x4(&a[j + 96], v3);
    }

    zl.val[0] = vld1q_s16(&invntt_br[k]);
    zh.val[0] = vld1q_s16(&invntt_qinv_br[k]);

#if FALCON_N == 512
    // Layer 7, 8
    for (j = 0; j < 64; j += 32)
    {
        vload_s16_x4(v0, &a[j]);
        vload_s16_x4(v1, &a[j + 128]);
        vload_s16_x4(v2, &a[j + 256]);
        vload_s16_x4(v3, &a[j + 384]);

        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);

        // v0: .5
        // v1: .5
        // v2: .5
        // v3: .5

        // Layer 7
        // v0 - v1, v2 - v3
        gsbf_bri_x4(v0, v1, zl.val[0], zh.val[0], 0, 0, 0, 0, neon_qmvq, t);
        gsbf_bri_x4(v2, v3, zl.val[0], zh.val[0], 1, 1, 1, 1, neon_qmvq, t);

        // v0: 1
        // v1: .87
        // v2: 1
        // v3: .87

        // Layer 8
        // v0 - v2, v1 - v3
        gsbf_bri_x4(v0, v2, zl.val[0], zh.val[0], 2, 2, 2, 2, neon_qmvq, t);
        gsbf_bri_x4(v1, v3, zl.val[0], zh.val[0], 2, 2, 2, 2, neon_qmvq, t);

        // v0: 2
        // v1: 1.75
        // v2: 1.25
        // v3: 1.25
        barmul_invntt_x4(v0, zl.val[0], zh.val[0], 3, neon_qmvq, t);
        barmul_invntt_x4(v1, zl.val[0], zh.val[0], 3, neon_qmvq, t);

        // v0: 1.25
        // v1: 1.15
        // v2: .97
        // v3: .97
        vstore_s16_x4(&a[j], v0);
        vstore_s16_x4(&a[j + 128], v1);
        vstore_s16_x4(&a[j + 256], v2);
        vstore_s16_x4(&a[j + 384], v3);
    }
    for (; j < 128; j += 32)
    {
        vload_s16_x4(v0, &a[j]);
        vload_s16_x4(v1, &a[j + 128]);
        vload_s16_x4(v2, &a[j + 256]);
        vload_s16_x4(v3, &a[j + 384]);

        // v0: 1.3
        // v1: 1.3
        // v2: 1.3
        // v3: 1.3

        // Layer 7
        // v0 - v1, v2 - v3
        gsbf_bri_x4(v0, v1, zl.val[0], zh.val[0], 0, 0, 0, 0, neon_qmvq, t);
        gsbf_bri_x4(v2, v3, zl.val[0], zh.val[0], 1, 1, 1, 1, neon_qmvq, t);

        // v0: 2.6
        // v1: 1.5
        // v2: 2.6
        // v3: 1.5

        barrett_x4(v0, neon_qmvq, t);
        barrett_x4(v1, neon_qmvq, t);
        barrett_x4(v2, neon_qmvq, t);
        barrett_x4(v3, neon_qmvq, t);

        // v0: 0.5
        // v1: 0.5
        // v2: 0.5
        // v3: 0.5

        // Layer 8
        // v0 - v2, v1 - v3
        gsbf_bri_x4(v0, v2, zl.val[0], zh.val[0], 2, 2, 2, 2, neon_qmvq, t);
        gsbf_bri_x4(v1, v3, zl.val[0], zh.val[0], 2, 2, 2, 2, neon_qmvq, t);

        // v0: 1
        // v1: 1
        // v2: .87
        // v3: .87
        barmul_invntt_x4(v0, zl.val[0], zh.val[0], 3, neon_qmvq, t);
        barmul_invntt_x4(v1, zl.val[0], zh.val[0], 3, neon_qmvq, t);

        // v0: .87
        // v1: .87
        // v2: .83
        // v3: .83

        vstore_s16_x4(&a[j], v0);
        vstore_s16_x4(&a[j + 128], v1);
        vstore_s16_x4(&a[j + 256], v2);
        vstore_s16_x4(&a[j + 384], v3);
    }
#elif FALCON_N == 1024
    // Layer 7, 8, 9
    int16x8x2_t u0, u1, u2, u3, u4, u5, u6, u7;

    for (j = 0; j < 64; j += 16)
    {
        vload_s16_x2(u0, &a[j]);
        vload_s16_x2(u1, &a[j + 128]);
        vload_s16_x2(u2, &a[j + 256]);
        vload_s16_x2(u3, &a[j + 384]);

        vload_s16_x2(u4, &a[j + 512]);
        vload_s16_x2(u5, &a[j + 640]);
        vload_s16_x2(u6, &a[j + 768]);
        vload_s16_x2(u7, &a[j + 896]);

        // 2
        barrett_x2(u0, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u1, 0, 1, 2, 3, neon_qmvq, t);
        barrett_x2(u2, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u3, 0, 1, 2, 3, neon_qmvq, t);

        barrett_x2(u4, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u5, 0, 1, 2, 3, neon_qmvq, t);
        barrett_x2(u6, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u7, 0, 1, 2, 3, neon_qmvq, t);

        // u0, 4: 0.5
        // u1, 5: 0.5
        // u2, 6: 0.5
        // u3, 7: 0.5

        // Layer 7
        // u0 - u1, u2 - u3
        // u4 - u5, u6 - u7
        gsbf_bri(u0.val[0], u1.val[0], zl.val[0], zh.val[0], 0, neon_qmvq, t.val[0]);
        gsbf_bri(u0.val[1], u1.val[1], zl.val[0], zh.val[0], 0, neon_qmvq, t.val[1]);
        gsbf_bri(u2.val[0], u3.val[0], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[2]);
        gsbf_bri(u2.val[1], u3.val[1], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[3]);

        gsbf_bri(u4.val[0], u5.val[0], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[0]);
        gsbf_bri(u4.val[1], u5.val[1], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[1]);
        gsbf_bri(u6.val[0], u7.val[0], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[2]);
        gsbf_bri(u6.val[1], u7.val[1], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[3]);

        // u0, 4: 1
        // u1, 5: .87
        // u2, 6: 1
        // u3, 7: .87

        // Layer 8
        // u0 - u2, u1 - u3
        // u4 - u6, u5 - u7
        gsbf_bri(u0.val[0], u2.val[0], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[0]);
        gsbf_bri(u0.val[1], u2.val[1], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[1]);
        gsbf_bri(u1.val[0], u3.val[0], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[2]);
        gsbf_bri(u1.val[1], u3.val[1], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[3]);

        gsbf_bri(u4.val[0], u6.val[0], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[0]);
        gsbf_bri(u4.val[1], u6.val[1], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[1]);
        gsbf_bri(u5.val[0], u7.val[0], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[2]);
        gsbf_bri(u5.val[1], u7.val[1], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[3]);

        // u0, 4: 2
        // u2, 6: 1.25
        // u1, 5: 1.75
        // u3, 7: 1.15

        barrett_x2(u0, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u4, 0, 1, 2, 3, neon_qmvq, t);
        barrett_x2(u1, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u5, 0, 1, 2, 3, neon_qmvq, t);

        // u0, 4: 0.5
        // u2, 6: 1.25
        // u1, 5: 0.5
        // u3, 7: 1.15

        // Layer 9
        // u0 - u4, u1 - u5
        // u2 - u6, u3 - u7
        gsbf_bri(u0.val[0], u4.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[0]);
        gsbf_bri(u0.val[1], u4.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[1]);
        gsbf_bri(u1.val[0], u5.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[2]);
        gsbf_bri(u1.val[1], u5.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[3]);

        gsbf_bri(u2.val[0], u6.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[0]);
        gsbf_bri(u2.val[1], u6.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[1]);
        gsbf_bri(u3.val[0], u7.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[2]);
        gsbf_bri(u3.val[1], u7.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[3]);

        barmul_invntt_x2(u0, zl.val[0], zh.val[0], 7, neon_qmvq, t);
        barmul_invntt_x2(u1, zl.val[0], zh.val[0], 7, neon_qmvq, t);
        barmul_invntt_x2(u2, zl.val[0], zh.val[0], 7, neon_qmvq, t);
        barmul_invntt_x2(u3, zl.val[0], zh.val[0], 7, neon_qmvq, t);

        vstore_s16_x2(&a[j], u0);
        vstore_s16_x2(&a[j + 128], u1);
        vstore_s16_x2(&a[j + 256], u2);
        vstore_s16_x2(&a[j + 384], u3);

        vstore_s16_x2(&a[j + 512], u4);
        vstore_s16_x2(&a[j + 640], u5);
        vstore_s16_x2(&a[j + 768], u6);
        vstore_s16_x2(&a[j + 896], u7);
    }

    for (; j < 128; j += 16)
    {
        vload_s16_x2(u0, &a[j]);
        vload_s16_x2(u1, &a[j + 128]);
        vload_s16_x2(u2, &a[j + 256]);
        vload_s16_x2(u3, &a[j + 384]);

        vload_s16_x2(u4, &a[j + 512]);
        vload_s16_x2(u5, &a[j + 640]);
        vload_s16_x2(u6, &a[j + 768]);
        vload_s16_x2(u7, &a[j + 896]);

        // 1.3

        // Layer 7
        // u0 - u1, u2 - u3
        // u4 - u5, u6 - u7
        gsbf_bri(u0.val[0], u1.val[0], zl.val[0], zh.val[0], 0, neon_qmvq, t.val[0]);
        gsbf_bri(u0.val[1], u1.val[1], zl.val[0], zh.val[0], 0, neon_qmvq, t.val[1]);
        gsbf_bri(u2.val[0], u3.val[0], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[2]);
        gsbf_bri(u2.val[1], u3.val[1], zl.val[0], zh.val[0], 1, neon_qmvq, t.val[3]);

        gsbf_bri(u4.val[0], u5.val[0], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[0]);
        gsbf_bri(u4.val[1], u5.val[1], zl.val[0], zh.val[0], 2, neon_qmvq, t.val[1]);
        gsbf_bri(u6.val[0], u7.val[0], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[2]);
        gsbf_bri(u6.val[1], u7.val[1], zl.val[0], zh.val[0], 3, neon_qmvq, t.val[3]);

        // u0, 4: 2.6
        // u1, 5: 1.5
        // u2, 6: 2.6
        // u3, 7: 1.5

        barrett_x2(u0, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u2, 0, 1, 2, 3, neon_qmvq, t);
        barrett_x2(u1, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u3, 0, 1, 2, 3, neon_qmvq, t);

        barrett_x2(u4, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u6, 0, 1, 2, 3, neon_qmvq, t);
        barrett_x2(u5, 0, 1, 0, 1, neon_qmvq, t);
        barrett_x2(u7, 0, 1, 2, 3, neon_qmvq, t);

        // u0, 4: .5
        // u1, 5: .5
        // u2, 6: .5
        // u3, 7: .5

        // Layer 8
        // u0 - u2, u1 - u3
        // u4 - u6, u5 - u7
        gsbf_bri(u0.val[0], u2.val[0], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[0]);
        gsbf_bri(u0.val[1], u2.val[1], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[1]);
        gsbf_bri(u1.val[0], u3.val[0], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[2]);
        gsbf_bri(u1.val[1], u3.val[1], zl.val[0], zh.val[0], 4, neon_qmvq, t.val[3]);

        gsbf_bri(u4.val[0], u6.val[0], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[0]);
        gsbf_bri(u4.val[1], u6.val[1], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[1]);
        gsbf_bri(u5.val[0], u7.val[0], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[2]);
        gsbf_bri(u5.val[1], u7.val[1], zl.val[0], zh.val[0], 5, neon_qmvq, t.val[3]);

        // u0, 4: 1
        // u2, 6: .87
        // u1, 5: 1
        // u3, 7: .87

        // Layer 9
        // u0 - u4, u1 - u5
        // u2 - u6, u3 - u7
        gsbf_bri(u0.val[0], u4.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[0]);
        gsbf_bri(u0.val[1], u4.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[1]);
        gsbf_bri(u1.val[0], u5.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[2]);
        gsbf_bri(u1.val[1], u5.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[3]);

        gsbf_bri(u2.val[0], u6.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[0]);
        gsbf_bri(u2.val[1], u6.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[1]);
        gsbf_bri(u3.val[0], u7.val[0], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[2]);
        gsbf_bri(u3.val[1], u7.val[1], zl.val[0], zh.val[0], 6, neon_qmvq, t.val[3]);

        // u0, 4: 2, 1.25
        // u2, 6: 1.75, 1.15
        // u1, 5: 2, 1.25
        // u3, 7: .175, 1.15

        barmul_invntt_x2(u0, zl.val[0], zh.val[0], 7, neon_qmvq, t);
        barmul_invntt_x2(u1, zl.val[0], zh.val[0], 7, neon_qmvq, t);
        barmul_invntt_x2(u2, zl.val[0], zh.val[0], 7, neon_qmvq, t);
        barmul_invntt_x2(u3, zl.val[0], zh.val[0], 7, neon_qmvq, t);

        vstore_s16_x2(&a[j], u0);
        vstore_s16_x2(&a[j + 128], u1);
        vstore_s16_x2(&a[j + 256], u2);
        vstore_s16_x2(&a[j + 384], u3);

        vstore_s16_x2(&a[j + 512], u4);
        vstore_s16_x2(&a[j + 640], u5);
        vstore_s16_x2(&a[j + 768], u6);
        vstore_s16_x2(&a[j + 896], u7);
    }

#else
#error "FALCON_N is either 512 or 1024"
#endif
}

/*
 * Reduce a small signed integer modulo q. The source integer MUST
 * be between -q/2 and +q/2.
 * TODO: remove this function
 */
extern inline uint32_t
mq_conv_small(int x)
{
    /*
     * If x < 0, the cast to uint32_t will set the high bit to 1.
     */
    uint32_t y;

    y = (uint32_t)x;
    y += Q & -(y >> 31);
    return y;
}
/*
 * Subtraction modulo q. Operands must be in the 0..q-1 range.
 */
extern inline uint32_t
mq_sub(uint32_t x, uint32_t y)
{
    /*
     * As in mq_add(), we use a conditional addition to ensure the
     * result is in the 0..q-1 range.
     */
    uint32_t d;

    d = x - y;
    d += Q & -(d >> 31);
    return d;
}

/*
 * Montgomery multiplication modulo q. If we set R = 2^16 mod q, then
 * this function computes: x * y / R mod q
 * Operands must be in the 0..q-1 range.
 */
static inline uint32_t
mq_montymul(uint32_t x, uint32_t y)
{
    uint32_t z, w;

    /*
     * We compute x*y + k*q with a value of k chosen so that the 16
     * low bits of the result are 0. We can then shift the value.
     * After the shift, result may still be larger than q, but it
     * will be lower than 2*q, so a conditional subtraction works.
     */

    z = x * y;
    w = ((z * Q0I) & 0xFFFF) * Q;

    /*
     * When adding z and w, the result will have its low 16 bits
     * equal to 0. Since x, y and z are lower than q, the sum will
     * be no more than (2^15 - 1) * q + (q - 1)^2, which will
     * fit on 29 bits.
     */
    z = (z + w) >> 16;

    /*
     * After the shift, analysis shows that the value will be less
     * than 2q. We do a subtraction then conditional subtraction to
     * ensure the result is in the expected range.
     */
    z -= Q;
    z += Q & -(z >> 31);
    return z;
}

/*
 * Montgomery squaring (computes (x^2)/R).
 */
static inline uint32_t
mq_montysqr(uint32_t x)
{
    return mq_montymul(x, x);
}

/*
 * Divide x by y modulo q = 12289.
 */
extern inline uint32_t
mq_div_12289(uint32_t x, uint32_t y)
{
    /*
     * We invert y by computing y^(q-2) mod q.
     *
     * We use the following addition chain for exponent e = 12287:
     *
     *   e0 = 1
     *   e1 = 2 * e0 = 2
     *   e2 = e1 + e0 = 3
     *   e3 = e2 + e1 = 5
     *   e4 = 2 * e3 = 10
     *   e5 = 2 * e4 = 20
     *   e6 = 2 * e5 = 40
     *   e7 = 2 * e6 = 80
     *   e8 = 2 * e7 = 160
     *   e9 = e8 + e2 = 163
     *   e10 = e9 + e8 = 323
     *   e11 = 2 * e10 = 646
     *   e12 = 2 * e11 = 1292
     *   e13 = e12 + e9 = 1455
     *   e14 = 2 * e13 = 2910
     *   e15 = 2 * e14 = 5820
     *   e16 = e15 + e10 = 6143
     *   e17 = 2 * e16 = 12286
     *   e18 = e17 + e0 = 12287
     *
     * Additions on exponents are converted to Montgomery
     * multiplications. We define all intermediate results as so
     * many local variables, and let the C compiler work out which
     * must be kept around.
     */
    uint32_t y0, y1, y2, y3, y4, y5, y6, y7, y8, y9;
    uint32_t y10, y11, y12, y13, y14, y15, y16, y17, y18;

    y0 = mq_montymul(y, R2);
    y1 = mq_montysqr(y0);
    y2 = mq_montymul(y1, y0);
    y3 = mq_montymul(y2, y1);
    y4 = mq_montysqr(y3);
    y5 = mq_montysqr(y4);
    y6 = mq_montysqr(y5);
    y7 = mq_montysqr(y6);
    y8 = mq_montysqr(y7);
    y9 = mq_montymul(y8, y2);
    y10 = mq_montymul(y9, y8);
    y11 = mq_montysqr(y10);
    y12 = mq_montysqr(y11);
    y13 = mq_montymul(y12, y9);
    y14 = mq_montysqr(y13);
    y15 = mq_montysqr(y14);
    y16 = mq_montymul(y15, y10);
    y17 = mq_montysqr(y16);
    y18 = mq_montymul(y17, y0);

    /*
     * Final multiplication with x, which is not in Montgomery
     * representation, computes the correct division result.
     */
    return mq_montymul(y18, x);
}

/*
 * Convert a polynomial (mod q) to Montgomery representation.
 */
void mq_poly_tomonty(uint16_t *f, unsigned logn)
{
    size_t u, n;

    n = (size_t)1 << logn;
    for (u = 0; u < n; u++)
    {
        f[u] = (uint16_t)mq_montymul(f[u], R2);
    }
}

/*
 * Multiply two polynomials together (NTT representation, and using
 * a Montgomery multiplication). Result f*g is written over f.
 */
void mq_poly_montymul_ntt(uint16_t *f, const uint16_t *g, unsigned logn)
{
    size_t u, n;

    n = (size_t)1 << logn;
    for (u = 0; u < n; u++)
    {
        f[u] = (uint16_t)mq_montymul(f[u], g[u]);
    }
}

/*
 * Subtract polynomial g from polynomial f.
 */
void mq_poly_sub(uint16_t *f, const uint16_t *g, unsigned logn)
{
    size_t u, n;

    n = (size_t)1 << logn;
    for (u = 0; u < n; u++)
    {
        f[u] = (uint16_t)mq_sub(f[u], g[u]);
    }
}

/* ===================================================================== */
