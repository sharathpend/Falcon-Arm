#include "sampler.h"
#include <arm_neon.h>

int sampler_neon(uint32_t v0, uint32_t v1, uint32_t v2)
{
    static const uint32_t dist[] = {
        10745844u, 3068844u, 3741698u,
        5559083u, 1580863u, 8248194u,
        2260429u, 13669192u, 2736639u,
        708981u, 4421575u, 10046180u,
        169348u, 7122675u, 4136815u,
        30538u, 13063405u, 7650655u,
        4132u, 14505003u, 7826148u,
        417u, 16768101u, 11363290u,
        31u, 8444042u, 8086568u,
        1u, 12844466u, 265321u,
        0u, 1232676u, 13644283u,
        0u, 38047u, 9111839u,
        0u, 870u, 6138264u,
        0u, 14u, 12545723u,
        0u, 0u, 3104126u,
        0u, 0u, 28824u,
        0u, 0u, 198u,
        0u, 0u, 1u};

    uint32x4x3_t w;
    uint32x4_t x0, x1, x2, cc0, cc1, cc2, zz;
    uint32x2x3_t wh;
    uint32x2_t cc0h, cc1h, cc2h, zzh;
    int z;
    x0 = vdupq_n_u32(v0);
    x1 = vdupq_n_u32(v1);
    x2 = vdupq_n_u32(v2);

    // 0: 0, 3, 6, 9
    // 1: 1, 4, 7, 10
    // 2: 2, 5, 8, 11
    // v0 - w0
    // v1 - w1
    // v2 - w2
    // cc1 - cc0 >> 31
    // cc2 - cc1 >> 31
    // z + cc2 >> 31
    w = vld3q_u32(&dist[0]);
    cc0 = vsubq_u32(x0, w.val[2]);
    cc1 = vsubq_u32(x1, w.val[1]);
    cc2 = vsubq_u32(x2, w.val[0]);
    cc1 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc1, (int32x4_t)cc0, 31);
    cc2 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc2, (int32x4_t)cc1, 31);
    zz = vshrq_n_u32(cc2, 31);

    w = vld3q_u32(&dist[12]);
    cc0 = vsubq_u32(x0, w.val[2]);
    cc1 = vsubq_u32(x1, w.val[1]);
    cc2 = vsubq_u32(x2, w.val[0]);
    cc1 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc1, (int32x4_t)cc0, 31);
    cc2 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc2, (int32x4_t)cc1, 31);
    zz = vsraq_n_u32(zz, cc2, 31);

    w = vld3q_u32(&dist[24]);
    cc0 = vsubq_u32(x0, w.val[2]);
    cc1 = vsubq_u32(x1, w.val[1]);
    cc2 = vsubq_u32(x2, w.val[0]);
    cc1 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc1, (int32x4_t)cc0, 31);
    cc2 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc2, (int32x4_t)cc1, 31);
    zz = vsraq_n_u32(zz, cc2, 31);

    w = vld3q_u32(&dist[36]);
    cc0 = vsubq_u32(x0, w.val[2]);
    cc1 = vsubq_u32(x1, w.val[1]);
    cc2 = vsubq_u32(x2, w.val[0]);
    cc1 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc1, (int32x4_t)cc0, 31);
    cc2 = (uint32x4_t)vsraq_n_s32((int32x4_t)cc2, (int32x4_t)cc1, 31);
    zz = vsraq_n_u32(zz, cc2, 31);

    // 0: 48, 51
    // 1: 49, 52
    // 2: 50, 53
    wh = vld3_u32(&dist[48]);
    cc0h = vsub_u32(vget_low_u32(x0), wh.val[2]);
    cc1h = vsub_u32(vget_low_u32(x1), wh.val[1]);
    cc2h = vsub_u32(vget_low_u32(x2), wh.val[0]);
    cc1h = (uint32x2_t)vsra_n_s32((int32x2_t)cc1h, (int32x2_t)cc0h, 31);
    cc2h = (uint32x2_t)vsra_n_s32((int32x2_t)cc2h, (int32x2_t)cc1h, 31);
    zzh = vshr_n_u32(cc2h, 31);

    z = vaddvq_u32(zz) + vaddv_u32(zzh);
    return z;
}

int sampler(uint32_t v0, uint32_t v1, uint32_t v2)
{
    static const uint32_t dist[] = {
        10745844u, 3068844u, 3741698u,
        5559083u, 1580863u, 8248194u,
        2260429u, 13669192u, 2736639u,
        708981u, 4421575u, 10046180u,
        169348u, 7122675u, 4136815u,
        30538u, 13063405u, 7650655u,
        4132u, 14505003u, 7826148u,
        417u, 16768101u, 11363290u,
        31u, 8444042u, 8086568u,
        1u, 12844466u, 265321u,
        0u, 1232676u, 13644283u,
        0u, 38047u, 9111839u,
        0u, 870u, 6138264u,
        0u, 14u, 12545723u,
        0u, 0u, 3104126u,
        0u, 0u, 28824u,
        0u, 0u, 198u,
        0u, 0u, 1u};

    int z;

    /*
     * Sampled value is z, such that v0..v2 is lower than the first
     * z elements of the table.
     */
    z = 0;
    for (int u = 0; u < (sizeof dist) / sizeof(dist[0]); u += 3)
    {
        uint32_t w0, w1, w2, cc;

        w0 = dist[u + 2];
        w1 = dist[u + 1];
        w2 = dist[u + 0];
        cc = (v0 - w0) >> 31;
        cc = (v1 - w1 - cc) >> 31;
        cc = (v2 - w2 - cc) >> 31;
        z += (int)cc;
    }
    return z;
}

#define TESTS (1 << 24)

int main()
{
    uint32_t v0, v1, v2;
    srand(0);
    int gold, test;
    for (int i = 0; i < TESTS; i++)
    {
        v0 = rand() & 0xFFFFFF;
        v1 = (v0 * 1337 + 124) & 0xFFFFFF;
        v2 = (v1 * 1337 + 124) & 0xFFFFFF;

        gold = sampler(v0, v1, v2);
        test = sampler_neon(v0, v1, v2);
        if (gold != test)
        {
            printf("%d: %d != %d\n", i, gold, test);
            return 1;
        }
    }
    return 0;
}