#include <arm_neon.h>
#include "../inner.h"
#include "../macrous.h"
#include "../config.h"
#include <stdio.h>
#include <assert.h>
#include "../macrof.h"

#define DEBUG 0

static const uint32_t l2bound[] = {
    0, /* unused */
    101498,
    208714,
    428865,
    892039,
    1852696,
    3842630,
    7959734,
    16468416,
    34034726,
    70265242};

/* see inner.h */
int Zf(is_short)(const int16_t *s1, const int16_t *s2, unsigned logn)
{
    /*
     * We use the l2-norm. Code below uses only 32-bit operations to
     * compute the square of the norm with saturation to 2^32-1 if
     * the value exceeds 2^31-1.
     */
    size_t n, u;
    uint32_t s, ng;

    n = (size_t)1 << logn;
    s = 0;
    ng = 0;
    for (u = 0; u < n; u++)
    {
        int32_t z;

        z = s1[u];
        s += (uint32_t)(z * z);
        ng |= s;
        z = s2[u];
        s += (uint32_t)(z * z);
        ng |= s;
    }
    s |= -(ng >> 31);

    printf("ref  %8x\n", s);

    return s <= l2bound[logn];
}

/* see inner.h */
int ZfN(is_short)(const int16_t *s1, const int16_t *s2)
{
    int16x8x4_t neon_s1, neon_s2;
    int32x4_t neon_s, neon_sh;
    uint32_t s;
    neon_s = vdupq_n_s32(0);
    neon_sh = vdupq_n_s32(0);

    for (unsigned u = 0; u < FALCON_N; u += 32)
    {
        vload_s16_x4(neon_s1, &s1[u]);

        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s1.val[0]), vget_low_s16(neon_s1.val[0]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s1.val[1]), vget_low_s16(neon_s1.val[1]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s1.val[2]), vget_low_s16(neon_s1.val[2]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s1.val[3]), vget_low_s16(neon_s1.val[3]));

        neon_sh = vqdmlal_high_s16(neon_sh, neon_s1.val[0], neon_s1.val[0]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_s1.val[1], neon_s1.val[1]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_s1.val[2], neon_s1.val[2]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_s1.val[3], neon_s1.val[3]);
    }
    for (unsigned u = 0; u < FALCON_N; u += 32)
    {
        vload_s16_x4(neon_s2, &s2[u]);

        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s2.val[0]), vget_low_s16(neon_s2.val[0]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s2.val[1]), vget_low_s16(neon_s2.val[1]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s2.val[2]), vget_low_s16(neon_s2.val[2]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_s2.val[3]), vget_low_s16(neon_s2.val[3]));

        neon_sh = vqdmlal_high_s16(neon_sh, neon_s2.val[0], neon_s2.val[0]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_s2.val[1], neon_s2.val[1]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_s2.val[2], neon_s2.val[2]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_s2.val[3], neon_s2.val[3]);
    }
    // 32x4
    neon_s = vhaddq_s32(neon_s, neon_sh);
    // 32x4 -> 32x1
    s = vaddvq_s32(neon_s);

    printf("neon %8x\n", s);

    return s <= l2bound[FALCON_LOGN];
}

/* see inner.h */
int Zf(is_short_half)(
    uint32_t sqn, const int16_t *s2, unsigned logn)
{
    size_t n, u;
    uint32_t ng;

    n = (size_t)1 << logn;
    ng = -(sqn >> 31);
    for (u = 0; u < n; u++)
    {
        int32_t z;

        z = s2[u];
        sqn += (uint32_t)(z * z);
        ng |= sqn;
    }
    sqn |= -(ng >> 31);

    return sqn <= l2bound[logn];
}

int ZfN(is_short_tmp)(int16_t s1tmp[FALCON_N], int16_t s2tmp[FALCON_N],
                       const int16_t hm[FALCON_N], const fpr t0[FALCON_N], 
                       const fpr t1[FALCON_N])
{
    int16x8x4_t neon_hm, neon_ts;
    float64x2x4_t neon_tf0, neon_tf1, neon_tf2, neon_tf3;
    int64x2x4_t neon_ts0, neon_ts1, neon_ts2, neon_ts3;
    int32x4x4_t neon_ts4, neon_ts5;
    int32x4_t neon_s, neon_sh;
    uint32_t s;

    neon_s = vdupq_n_s32(0);
    neon_sh = vdupq_n_s32(0);

    // s1tmp
    for (int i = 0; i < FALCON_N; i += 32)
    {
        vloadx4(neon_tf0, &t0[i]);
        vloadx4(neon_tf1, &t0[i + 8]);
        vloadx4(neon_tf2, &t0[i + 16]);
        vloadx4(neon_tf3, &t0[i + 24]);
        vload_s16_x4(neon_hm, &hm[i]);

        vfrintx4(neon_ts0, neon_tf0);
        vfrintx4(neon_ts1, neon_tf1);
        vfrintx4(neon_ts2, neon_tf2);
        vfrintx4(neon_ts3, neon_tf3);

        neon_ts4.val[0] = vmovn_high_s64(vmovn_s64(neon_ts0.val[0]), neon_ts0.val[1]);
        neon_ts4.val[1] = vmovn_high_s64(vmovn_s64(neon_ts0.val[2]), neon_ts0.val[3]);
        neon_ts4.val[2] = vmovn_high_s64(vmovn_s64(neon_ts1.val[0]), neon_ts1.val[1]);
        neon_ts4.val[3] = vmovn_high_s64(vmovn_s64(neon_ts1.val[2]), neon_ts1.val[3]);

        neon_ts5.val[0] = vmovn_high_s64(vmovn_s64(neon_ts2.val[0]), neon_ts2.val[1]);
        neon_ts5.val[1] = vmovn_high_s64(vmovn_s64(neon_ts2.val[2]), neon_ts2.val[3]);
        neon_ts5.val[2] = vmovn_high_s64(vmovn_s64(neon_ts3.val[0]), neon_ts3.val[1]);
        neon_ts5.val[3] = vmovn_high_s64(vmovn_s64(neon_ts3.val[2]), neon_ts3.val[3]);

        neon_ts.val[0] = vmovn_high_s32(vmovn_s32(neon_ts4.val[0]), neon_ts4.val[1]);
        neon_ts.val[1] = vmovn_high_s32(vmovn_s32(neon_ts4.val[2]), neon_ts4.val[3]);
        neon_ts.val[2] = vmovn_high_s32(vmovn_s32(neon_ts5.val[0]), neon_ts5.val[1]);
        neon_ts.val[3] = vmovn_high_s32(vmovn_s32(neon_ts5.val[2]), neon_ts5.val[3]);

        // hm = hm - fpr_rint(t0)
        neon_hm.val[0] = vsubq_s16(neon_hm.val[0], neon_ts.val[0]);
        neon_hm.val[1] = vsubq_s16(neon_hm.val[1], neon_ts.val[1]);
        neon_hm.val[2] = vsubq_s16(neon_hm.val[2], neon_ts.val[2]);
        neon_hm.val[3] = vsubq_s16(neon_hm.val[3], neon_ts.val[3]);

        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_hm.val[0]), vget_low_s16(neon_hm.val[0]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_hm.val[1]), vget_low_s16(neon_hm.val[1]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_hm.val[2]), vget_low_s16(neon_hm.val[2]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_hm.val[3]), vget_low_s16(neon_hm.val[3]));

        neon_sh = vqdmlal_high_s16(neon_sh, neon_hm.val[0], neon_hm.val[0]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_hm.val[1], neon_hm.val[1]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_hm.val[2], neon_hm.val[2]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_hm.val[3], neon_hm.val[3]);

        vstore_s16_x4(&s1tmp[i], neon_hm);
    }

    // s2tmp
    for (int i = 0; i < FALCON_N; i += 32)
    {
        vloadx4(neon_tf0, &t1[i]);
        vloadx4(neon_tf1, &t1[i + 8]);
        vloadx4(neon_tf2, &t1[i + 16]);
        vloadx4(neon_tf3, &t1[i + 24]);

        vfrintx4(neon_ts0, neon_tf0);
        vfrintx4(neon_ts1, neon_tf1);
        vfrintx4(neon_ts2, neon_tf2);
        vfrintx4(neon_ts3, neon_tf3);

        neon_ts4.val[0] = vmovn_high_s64(vmovn_s64(neon_ts0.val[0]), neon_ts0.val[1]);
        neon_ts4.val[1] = vmovn_high_s64(vmovn_s64(neon_ts0.val[2]), neon_ts0.val[3]);
        neon_ts4.val[2] = vmovn_high_s64(vmovn_s64(neon_ts1.val[0]), neon_ts1.val[1]);
        neon_ts4.val[3] = vmovn_high_s64(vmovn_s64(neon_ts1.val[2]), neon_ts1.val[3]);

        neon_ts5.val[0] = vmovn_high_s64(vmovn_s64(neon_ts2.val[0]), neon_ts2.val[1]);
        neon_ts5.val[1] = vmovn_high_s64(vmovn_s64(neon_ts2.val[2]), neon_ts2.val[3]);
        neon_ts5.val[2] = vmovn_high_s64(vmovn_s64(neon_ts3.val[0]), neon_ts3.val[1]);
        neon_ts5.val[3] = vmovn_high_s64(vmovn_s64(neon_ts3.val[2]), neon_ts3.val[3]);

        neon_ts.val[0] = vmovn_high_s32(vmovn_s32(neon_ts4.val[0]), neon_ts4.val[1]);
        neon_ts.val[1] = vmovn_high_s32(vmovn_s32(neon_ts4.val[2]), neon_ts4.val[3]);
        neon_ts.val[2] = vmovn_high_s32(vmovn_s32(neon_ts5.val[0]), neon_ts5.val[1]);
        neon_ts.val[3] = vmovn_high_s32(vmovn_s32(neon_ts5.val[2]), neon_ts5.val[3]);

        neon_ts.val[0] = vnegq_s16(neon_ts.val[0]);
        neon_ts.val[1] = vnegq_s16(neon_ts.val[1]);
        neon_ts.val[2] = vnegq_s16(neon_ts.val[2]);
        neon_ts.val[3] = vnegq_s16(neon_ts.val[3]);
        vstore_s16_x4(&s2tmp[i], neon_ts);

        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_ts.val[0]), vget_low_s16(neon_ts.val[0]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_ts.val[1]), vget_low_s16(neon_ts.val[1]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_ts.val[2]), vget_low_s16(neon_ts.val[2]));
        neon_s = vqdmlal_s16(neon_s, vget_low_s16(neon_ts.val[3]), vget_low_s16(neon_ts.val[3]));

        neon_sh = vqdmlal_high_s16(neon_sh, neon_ts.val[0], neon_ts.val[0]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_ts.val[1], neon_ts.val[1]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_ts.val[2], neon_ts.val[2]);
        neon_sh = vqdmlal_high_s16(neon_sh, neon_ts.val[3], neon_ts.val[3]);
    }

    // 32x4
    neon_s = vhaddq_s32(neon_s, neon_sh);
    // 32x4 -> 32x1
    s = vaddvq_s32(neon_s);

    return s <= l2bound[FALCON_LOGN];
}

#define TESTS 100

void center_q(int16_t *a)
{
    for (int i = 0; i < FALCON_N; i++)
    {
        if (a[i] > FALCON_Q / 2)
        {
            a[i] -= FALCON_Q;
        }
        else if (a[i] < -FALCON_Q / 2)
        {
            a[i] += FALCON_Q;
        }

        // assert (a[i] <= FALCON_Q/2);
        // assert (a[i] >= -FALCON_Q/2);
    }
}

int main()
{
    int16_t s1[FALCON_N], s2[FALCON_N];
    int test, gold, ret = 0;

    for (int t = 0; t < TESTS; t++)
    {
        for (int i = 0; i < FALCON_N; i++)
        {
            s1[i] = rand() % FALCON_Q;
            s2[i] = rand() % FALCON_Q;
        }
        center_q(s1);
        center_q(s2);

        gold = Zf(is_short)(s1, s2, FALCON_LOGN);
        test = ZfN(is_short)(s1, s2);

        // printf("test, gold: %d -- %d\n", test, gold);
        if (test != gold)
        {
            return 1;
        }

        printf("iter %d\n", t);
    }

    return 0;
}
