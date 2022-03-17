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

void neon_conv_small(int16_t out[FALCON_N], const int8_t in[FALCON_N])
{
    // Total SIMD registers: 24 = 16 + 8
    int16x8x4_t a, b, e, f; // 16
    int8x16x4_t c, d;       // 8

    for (int i = 0; i < FALCON_N; i += 128)
    {
        c = vld1q_s8_x4(&in[i]);
        d = vld1q_s8_x4(&in[i + 64]);

        a.val[0] = vmovl_s8(vget_low_s8(c.val[0]));
        a.val[2] = vmovl_s8(vget_low_s8(c.val[1]));
        b.val[0] = vmovl_s8(vget_low_s8(c.val[2]));
        b.val[2] = vmovl_s8(vget_low_s8(c.val[3]));

        a.val[1] = vmovl_high_s8(c.val[0]);
        a.val[3] = vmovl_high_s8(c.val[1]);
        b.val[1] = vmovl_high_s8(c.val[2]);
        b.val[3] = vmovl_high_s8(c.val[3]);

        e.val[0] = vmovl_s8(vget_low_s8(d.val[0]));
        e.val[2] = vmovl_s8(vget_low_s8(d.val[1]));
        f.val[0] = vmovl_s8(vget_low_s8(d.val[2]));
        f.val[2] = vmovl_s8(vget_low_s8(d.val[3]));

        e.val[1] = vmovl_high_s8(d.val[0]);
        e.val[3] = vmovl_high_s8(d.val[1]);
        f.val[1] = vmovl_high_s8(d.val[2]);
        f.val[3] = vmovl_high_s8(d.val[3]);

        vst1q_s16_x4(&out[i], a);
        vst1q_s16_x4(&out[i + 32], b);
        vst1q_s16_x4(&out[i + 64], e);
        vst1q_s16_x4(&out[i + 96], f);
    }
}

/*
 * Return f[] = f[]/g[] % 12289
 * See assembly https://godbolt.org/z/G59vo4crY
 */

void neon_div_12289(int16_t f[FALCON_N], const int16_t g[FALCON_N])
{
    // Total SIMD registers: 24 = 4 + 19 + 1
    int16x8x4_t src, dst, t, k; // 4
    int16x8x4_t y0, y1, y2, y3, y4, y5,
        y6, y7, y8, y9, y10, y11, y12,
        y13, y14, y15, y16, y17, y18; // 19
    int16x8_t neon_qmvm;              // 1

    neon_qmvm = vld1q_s16(qmvq);

    for (int i = 0; i < FALCON_N; i += 32)
    {
        // Find y0 = g^12287
        vload_s16_x4(y0, &g[i]);
        vload_s16_x4(src, &f[i]);

        // y0 = y0 * mont
        barmuli_mont_x4(y0, neon_qmvm, k);

        montmul_x4(y1, y0, y0, neon_qmvm, t);
        montmul_x4(y2, y1, y0, neon_qmvm, k);
        montmul_x4(y3, y2, y1, neon_qmvm, t);
        montmul_x4(y4, y3, y3, neon_qmvm, k);
        montmul_x4(y5, y4, y4, neon_qmvm, t);
        montmul_x4(y6, y5, y5, neon_qmvm, k);
        montmul_x4(y7, y6, y6, neon_qmvm, t);
        montmul_x4(y8, y7, y7, neon_qmvm, k);
        montmul_x4(y9, y8, y2, neon_qmvm, t);
        montmul_x4(y10, y9, y8, neon_qmvm, k);
        montmul_x4(y11, y10, y10, neon_qmvm, t);
        montmul_x4(y12, y11, y11, neon_qmvm, k);
        montmul_x4(y13, y12, y9, neon_qmvm, t);
        montmul_x4(y14, y13, y13, neon_qmvm, k);
        montmul_x4(y15, y14, y14, neon_qmvm, t);
        montmul_x4(y16, y15, y10, neon_qmvm, k);
        montmul_x4(y17, y16, y16, neon_qmvm, t);
        montmul_x4(y18, y17, y0, neon_qmvm, k);
        montmul_x4(dst, y18, src, neon_qmvm, t);

        vstore_s16_x4(&f[i], dst);
    }
}

void neon_poly_montymul_ntt(int16_t *f, const int16_t *g)
{
    // Total SIMD registers: 29 = 28 + 1
    int16x8x4_t a, b, c, d, e1, e2, t, k; // 28
    int16x8_t neon_qmvm;                  // 1
    neon_qmvm = vld1q_s16(qmvq);

    for (int i = 0; i < FALCON_N; i += 64)
    {
        vload_s16_x4(a, &f[i]);
        vload_s16_x4(b, &g[i]);
        vload_s16_x4(c, &f[i + 32]);
        vload_s16_x4(d, &g[i + 32]);

        montmul_x4(e1, a, b, neon_qmvm, t);
        montmul_x4(e2, c, d, neon_qmvm, k);

        vstore_s16_x4(&f[i], e1);
        vstore_s16_x4(&f[i + 32], e2);
    }
}

void neon_poly_sub_barrett(int16_t *f, const int16_t *g)
{
    // Total SIMD registers: 29 = 28 + 1
    int16x8x4_t a, b, c, d, e, h, t; // 28
    int16x8_t neon_qmvm;             // 1
    neon_qmvm = vld1q_s16(qmvq);

    for (int i = 0; i < FALCON_N; i += 64)
    {
        vload_s16_x4(a, &f[i]);
        vload_s16_x4(b, &g[i]);
        vload_s16_x4(c, &f[i + 32]);
        vload_s16_x4(d, &g[i + 32]);

        e.val[0] = vsubq_s16(a.val[0], b.val[0]);
        e.val[1] = vsubq_s16(a.val[1], b.val[1]);
        e.val[2] = vsubq_s16(a.val[2], b.val[2]);
        e.val[3] = vsubq_s16(a.val[3], b.val[3]);

        h.val[0] = vsubq_s16(c.val[0], d.val[0]);
        h.val[1] = vsubq_s16(c.val[1], d.val[1]);
        h.val[2] = vsubq_s16(c.val[2], d.val[2]);
        h.val[3] = vsubq_s16(c.val[3], d.val[3]);

        barrett_x4(e, neon_qmvm, t);
        barrett_x4(h, neon_qmvm, t);

        vstore_s16_x4(&f[i], e);
        vstore_s16_x4(&f[i + 32], h);
    }
}

/*
 * Check f[] has 0
 * Return:
 * 1 if 0 in f[]
 * otherwise, 0
 */
uint16_t neon_compare_with_zero(int16_t f[FALCON_N])
{
    // Total SIMD registers: 22 = 12 + 8 + 2
    int16x8x4_t a, b;      // 8
    uint16x8x4_t c, d, e1; // 12
    uint16x8x2_t e2;       // 2

    e2.val[1] = vdupq_n_u16(0);

    for (int i = 0; i < FALCON_N; i += 64)
    {
        vload_s16_x4(a, &f[i]);
        vload_s16_x4(b, &f[i + 32]);

        // Compare bitwise Equal to zero (vector)
        // a == 0 ? 1 : 0;
        c.val[0] = vceqzq_s16(a.val[0]);
        c.val[1] = vceqzq_s16(a.val[1]);
        c.val[2] = vceqzq_s16(a.val[2]);
        c.val[3] = vceqzq_s16(a.val[3]);

        d.val[0] = vceqzq_s16(b.val[0]);
        d.val[1] = vceqzq_s16(b.val[1]);
        d.val[2] = vceqzq_s16(b.val[2]);
        d.val[3] = vceqzq_s16(b.val[3]);

        e1.val[0] = vorrq_u16(d.val[0], c.val[0]);
        e1.val[1] = vorrq_u16(d.val[1], c.val[1]);
        e1.val[2] = vorrq_u16(d.val[2], c.val[2]);
        e1.val[3] = vorrq_u16(d.val[3], c.val[3]);

        e1.val[0] = vorrq_u16(e1.val[0], e1.val[2]);
        e1.val[1] = vorrq_u16(e1.val[1], e1.val[3]);

        e2.val[0] = vorrq_u16(e1.val[0], e1.val[1]);

        e2.val[1] = vorrq_u16(e2.val[1], e2.val[0]);
    }

    uint16_t ret = vmaxvq_u16(e2.val[1]);

    return ret;
}

/*
 * Return h = c0 - s1
 */
void neon_poly_sub(int16_t h[FALCON_N], const int16_t c0[FALCON_N], const int16_t s1[FALCON_N])
{
    // Total SIMD registers: 24
    int16x8x4_t a[2], b[2], c[2]; // 24

    for (int i = 0; i < FALCON_N; i += 64)
    {
        vload_s16_x4(a[0], &c0[i]);
        vload_s16_x4(a[1], &c0[i + 32]);

        vload_s16_x4(b[0], &s1[i]);
        vload_s16_x4(b[1], &s1[i + 32]);

        vsub_x4(c[0], a[0], b[0]);
        vsub_x4(c[1], a[1], b[1]);

        vstore_s16_x4(&h[i], c[0]);
        vstore_s16_x4(&h[i + 32], c[1]);
    }
}

/*
 * Branchless conditional addtion with FALCON_Q if coeffcient is < 0
 */
void neon_poly_unsigned(int16_t f[FALCON_N])
{
    // Total SIMD registers: 25 = 8 + 16 + 1
    uint16x8x4_t b[2];      // 8
    int16x8x4_t a[2], c[2]; // 16
    uint16x8_t neon_q;      // 1

    neon_q = vdupq_n_u16(FALCON_Q);

    for (int i = 0; i < FALCON_N; i += 64)
    {
        vload_s16_x4(a[0], &f[i]);
        vload_s16_x4(a[1], &f[i + 32]);

        b[0].val[0] = vcltzq_s16(a[0].val[0]);
        b[0].val[1] = vcltzq_s16(a[0].val[1]);
        b[0].val[2] = vcltzq_s16(a[0].val[2]);
        b[0].val[3] = vcltzq_s16(a[0].val[3]);

        b[1].val[0] = vcltzq_s16(a[1].val[0]);
        b[1].val[1] = vcltzq_s16(a[1].val[1]);
        b[1].val[2] = vcltzq_s16(a[1].val[2]);
        b[1].val[3] = vcltzq_s16(a[1].val[3]);

        c[0].val[0] = (int16x8_t)vandq_u16(b[0].val[0], neon_q);
        c[0].val[1] = (int16x8_t)vandq_u16(b[0].val[1], neon_q);
        c[0].val[2] = (int16x8_t)vandq_u16(b[0].val[2], neon_q);
        c[0].val[3] = (int16x8_t)vandq_u16(b[0].val[3], neon_q);

        c[1].val[0] = (int16x8_t)vandq_u16(b[1].val[0], neon_q);
        c[1].val[1] = (int16x8_t)vandq_u16(b[1].val[1], neon_q);
        c[1].val[2] = (int16x8_t)vandq_u16(b[1].val[2], neon_q);
        c[1].val[3] = (int16x8_t)vandq_u16(b[1].val[3], neon_q);

        vadd_x4(c[0], a[0], c[0]);
        vadd_x4(c[1], a[1], c[1]);

        vstore_s16_x4(&f[i], c[0]);
        vstore_s16_x4(&f[i + 32], c[1]);
    }
}

int neon_big_to_smallints(int8_t G[FALCON_N], const int16_t t[FALCON_N])
{
    // Total SIMD registers: 32
    int16x8x4_t a, f;              // 8
    uint16x8x4_t c[2], d[2];       // 16
    uint16x8x2_t e;                // 2
    int8x16x4_t g;                 // 4
    int16x8_t neon_127, neon__127; // 2
    neon_127 = vdupq_n_s16(127);
    neon__127 = vdupq_n_s16(-127);

    e.val[1] = vdupq_n_u16(0);

    for (int i = 0; i < FALCON_N; i += 64)
    {
        vload_s16_x4(a, &t[i]);
        vload_s16_x4(f, &t[i + 32]);

        g.val[0] = vmovn_high_s16(vmovn_s16(a.val[0]), a.val[1]);
        g.val[1] = vmovn_high_s16(vmovn_s16(a.val[2]), a.val[3]);
        g.val[2] = vmovn_high_s16(vmovn_s16(f.val[0]), f.val[1]);
        g.val[3] = vmovn_high_s16(vmovn_s16(f.val[2]), f.val[3]);

        vst1q_s8_x4(&G[i], g);

        // -127 > a ? 1 : 0
        c[0].val[0] = vcgtq_s16(neon__127, a.val[0]);
        c[0].val[1] = vcgtq_s16(neon__127, a.val[1]);
        c[0].val[2] = vcgtq_s16(neon__127, a.val[2]);
        c[0].val[3] = vcgtq_s16(neon__127, a.val[3]);
        // a > 127 ? 1 : 0
        c[1].val[0] = vcgtq_s16(a.val[0], neon_127);
        c[1].val[1] = vcgtq_s16(a.val[1], neon_127);
        c[1].val[2] = vcgtq_s16(a.val[2], neon_127);
        c[1].val[3] = vcgtq_s16(a.val[3], neon_127);

        // -127 > f ? 1 : 0
        d[0].val[0] = vcgtq_s16( neon__127, f.val[0]);
        d[0].val[1] = vcgtq_s16( neon__127, f.val[1]);
        d[0].val[2] = vcgtq_s16( neon__127, f.val[2]);
        d[0].val[3] = vcgtq_s16( neon__127, f.val[3]);
        // f > 127 ? 1 : 0
        d[1].val[0] = vcgtq_s16(f.val[0], neon_127);
        d[1].val[1] = vcgtq_s16(f.val[1], neon_127);
        d[1].val[2] = vcgtq_s16(f.val[2], neon_127);
        d[1].val[3] = vcgtq_s16(f.val[3], neon_127);

        c[0].val[0] = vorrq_u16(c[0].val[0], c[1].val[0]);
        c[0].val[1] = vorrq_u16(c[0].val[1], c[1].val[1]);
        c[0].val[2] = vorrq_u16(c[0].val[2], c[1].val[2]);
        c[0].val[3] = vorrq_u16(c[0].val[3], c[1].val[3]);

        d[0].val[0] = vorrq_u16(d[0].val[0], d[1].val[0]);
        d[0].val[1] = vorrq_u16(d[0].val[1], d[1].val[1]);
        d[0].val[2] = vorrq_u16(d[0].val[2], d[1].val[2]);
        d[0].val[3] = vorrq_u16(d[0].val[3], d[1].val[3]);

        c[0].val[0] = vorrq_u16(c[0].val[0], d[0].val[0]);
        c[0].val[2] = vorrq_u16(c[0].val[2], d[0].val[2]);
        c[0].val[1] = vorrq_u16(c[0].val[1], d[0].val[1]);
        c[0].val[3] = vorrq_u16(c[0].val[3], d[0].val[3]);

        c[0].val[0] = vorrq_u16(c[0].val[0], c[0].val[2]);
        c[0].val[1] = vorrq_u16(c[0].val[1], c[0].val[3]);

        e.val[0] = vorrq_u16(c[0].val[0], c[0].val[1]);

        e.val[1] = vorrq_u16(e.val[1], e.val[0]);
    }
    if (vmaxvq_u16(e.val[1]))
    {
        return 1;
    }
    return 0;
}

/* ===================================================================== */
