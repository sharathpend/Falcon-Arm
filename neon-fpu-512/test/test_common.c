#include <arm_neon.h>
#include "inner.h"
#include "macrous.h"
#include <stdio.h>

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
int
    ZfN(is_short)(
        const int16_t *s1, const int16_t *s2, const unsigned logn)
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
#if DEBUG
    printf("is_short ref  ng-s: %u - %u\n", ng, s);
    return s;
#endif
    s |= -(ng >> 31);
    return s <= l2bound[logn];
}


/* see inner.h */
int ZfN(is_short_half)(uint32_t sqn, const int16_t *s2, const unsigned logn)
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
    printf("is_short_half ref  sqn: %8x\n", sqn);
    sqn |= -(ng >> 31);

    printf("is_short_half ref  ng, sqn: %8x | %8x\n", ng, sqn);

    return sqn <= l2bound[logn];
}

int Zf(neon_is_short_half1)(uint32_t sqn, const int16_t *s2, const unsigned logn)
{
    int16x8x4_t s2_s16;
    int32x4_t neon_sqn, neon_zero;
    uint32x4_t neon_ng;
    const unsigned falcon_n = 1 << logn;
    uint32_t ng = -(sqn >> 31);

    neon_sqn = vdupq_n_s32(0);
    neon_zero = vdupq_n_s32(0);
    neon_ng = vdupq_n_u32(0);

    for (unsigned u = 0; u < falcon_n; u += 32)
    {
        s2_s16 = vld1q_s16_x4(&s2[u]);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[0], s2_s16.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vmulla_hi(neon_sqn, neon_sqn, s2_s16.val[0], s2_s16.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[1], s2_s16.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vmulla_hi(neon_sqn, neon_sqn, s2_s16.val[1], s2_s16.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[2], s2_s16.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vmulla_hi(neon_sqn, neon_sqn, s2_s16.val[2], s2_s16.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[3], s2_s16.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vmulla_hi(neon_sqn, neon_sqn, s2_s16.val[3], s2_s16.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
    }
    // 32x2
    neon_sqn = vpaddq_s32(neon_sqn, neon_zero);
    vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
    // ng |= sqn;
    sqn += vaddvq_s32(neon_sqn);
    ng |= sqn;
    ng |= vgetq_lane_u32(neon_ng, 0);
    ng |= vgetq_lane_u32(neon_ng, 1);
    ng |= vgetq_lane_u32(neon_ng, 2);
    ng |= vgetq_lane_u32(neon_ng, 3);

    printf("is_short_half neon sqn: %8x\n", sqn);

    sqn |= -(ng >> 31);

    printf("is_short_half neon ng, sqn: %8x | %8x\n", ng, sqn);

    return sqn <= l2bound[logn];
}

int Zf(neon_is_short)(const int16_t *s1, const int16_t *s2, unsigned logn)
{
    int16x8x4_t neon_s1, neon_s2;
    uint32x4_t neon_ng;
    int32x4_t neon_s, neon_zero;
    const unsigned falcon_n = 1 << logn;
    uint32_t s, ng;
    neon_s = vdupq_n_s32(0);
    neon_zero = vdupq_n_s32(0);
    neon_ng = vdupq_n_u32(0);

    for (unsigned u = 0; u < falcon_n; u += 32)
    {
        neon_s1 = vld1q_s16_x4(&s1[u]);
        neon_s2 = vld1q_s16_x4(&s2[u]);

        vmulla_lo(neon_s, neon_s, neon_s1.val[0], neon_s1.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s1.val[0], neon_s1.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);

        vmulla_lo(neon_s, neon_s, neon_s1.val[1], neon_s1.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s1.val[1], neon_s1.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);

        vmulla_lo(neon_s, neon_s, neon_s1.val[2], neon_s1.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s1.val[2], neon_s1.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);

        vmulla_lo(neon_s, neon_s, neon_s1.val[3], neon_s1.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s1.val[3], neon_s1.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        // 
        vmulla_lo(neon_s, neon_s, neon_s2.val[0], neon_s2.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s2.val[0], neon_s2.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);

        vmulla_lo(neon_s, neon_s, neon_s2.val[1], neon_s2.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s2.val[1], neon_s2.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);

        vmulla_lo(neon_s, neon_s, neon_s2.val[2], neon_s2.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s2.val[2], neon_s2.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);

        vmulla_lo(neon_s, neon_s, neon_s2.val[3], neon_s2.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
        vmulla_hi(neon_s, neon_s, neon_s2.val[3], neon_s2.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
    }
    // 32x2
    neon_s = vpaddq_s32(neon_s, neon_zero);
    vor(neon_ng, neon_ng, (uint32x4_t) neon_s);
    s = vaddvq_s32(neon_s);

    ng = vgetq_lane_u32(neon_ng, 0);
    ng |= vgetq_lane_u32(neon_ng, 1);
    ng |= vgetq_lane_u32(neon_ng, 2);
    ng |= vgetq_lane_u32(neon_ng, 3);
    ng |= s;

    printf("s: %8x\n", s);

    s |= -(ng >> 31);

    printf("neon s, ng: %8x | %8x\n", s, ng);

    return s <= l2bound[logn];

}

#define TESTS 100000
#define LOGN 10

int main()
{
    int16_t s1[1 << LOGN];
    int16_t tmp, tmp2;
    uint32_t sqn;
    int test, gold, ret = 0;

    sqn = 13884418;
    int16_t s2[] = {
        135,
        -406,
        143,
        158,
        280,
        -34,
        -233,
        39,
        -165,
        -357,
        129,
        -189,
        86,
        -350,
        -118,
        4,
        -40,
        -140,
        5,
        -96,
        94,
        41,
        -51,
        -89,
        -311,
        107,
        1,
        -213,
        -228,
        -25,
        315,
        -212,
        -106,
        215,
        61,
        -19,
        112,
        -53,
        -8,
        50,
        -181,
        135,
        102,
        -89,
        -193,
        115,
        -59,
        27,
        81,
        57,
        -260,
        -123,
        76,
        -255,
        190,
        73,
        18,
        -262,
        -216,
        -77,
        272,
        -180,
        101,
        -30,
        177,
        40,
        212,
        -110,
        -448,
        248,
        23,
        15,
        13,
        -171,
        -261,
        19,
        -124,
        97,
        -136,
        364,
        -82,
        74,
        339,
        -114,
        -308,
        -68,
        -2,
        -234,
        166,
        82,
        211,
        -135,
        76,
        -95,
        -190,
        395,
        -15,
        -249,
        87,
        29,
        -81,
        204,
        -231,
        -181,
        234,
        -145,
        -70,
        -222,
        238,
        356,
        10,
        50,
        48,
        222,
        123,
        83,
        -97,
        269,
        88,
        -12,
        124,
        -113,
        -335,
        73,
        -225,
        -251,
        -12,
        -89,
        45,
        167,
        -72,
        212,
        10,
        -117,
        147,
        109,
        -252,
        -289,
        -54,
        -274,
        -81,
        -159,
        1,
        77,
        244,
        -95,
        -124,
        70,
        -202,
        35,
        -166,
        -9,
        -60,
        157,
        38,
        -44,
        -109,
        -19,
        -137,
        37,
        80,
        -3,
        183,
        176,
        70,
        24,
        -108,
        170,
        -241,
        -72,
        151,
        2,
        -26,
        257,
        20,
        -66,
        -27,
        251,
        5,
        -1,
        13,
        355,
        178,
        69,
        -181,
        -71,
        32,
        198,
        128,
        -215,
        268,
        -196,
        44,
        132,
        -28,
        -86,
        33,
        -153,
        98,
        460,
        70,
        -4,
        34,
        14,
        -86,
        181,
        -178,
        35,
        -676,
        -86,
        -302,
        -177,
        200,
        85,
        381,
        125,
        -64,
        95,
        -28,
        22,
        71,
        -70,
        -289,
        -76,
        -132,
        -261,
        -50,
        -105,
        101,
        75,
        150,
        -238,
        6,
        131,
        -10,
        -3,
        -347,
        48,
        334,
        -105,
        32,
        25,
        -239,
        70,
        -160,
        52,
        125,
        59,
        94,
        217,
        -254,
        -62,
        -146,
        151,
        2,
        -188,
        -176,
        348,
        -174,
        71,
        74,
        -64,
        281,
        -3,
        -222,
        246,
        95,
        -67,
        -36,
        -202,
        238,
        -185,
        117,
        -146,
        -73,
        177,
        -84,
        -155,
        252,
        -179,
        -7,
        131,
        -24,
        -175,
        213,
        -45,
        39,
        -87,
        -50,
        -231,
        224,
        108,
        113,
        112,
        -131,
        118,
        -31,
        44,
        171,
        73,
        13,
        -44,
        79,
        276,
        172,
        207,
        197,
        252,
        215,
        -192,
        -151,
        68,
        -238,
        205,
        -183,
        -55,
        -108,
        -76,
        119,
        -14,
        -250,
        158,
        222,
        149,
        154,
        100,
        -184,
        272,
        -20,
        144,
        79,
        -174,
        113,
        120,
        -338,
        107,
        -82,
        -167,
        -55,
        -69,
        289,
        151,
        -37,
        32,
        125,
        -335,
        18,
        -18,
        33,
        208,
        -161,
        446,
        -172,
        303,
        -47,
        221,
        37,
        -177,
        102,
        51,
        -120,
        62,
        -79,
        -16,
        118,
        -236,
        -64,
        0,
        -87,
        -102,
        -112,
        -25,
        278,
        196,
        -244,
        -3,
        -270,
        -311,
        224,
        8,
        12,
        112,
        -103,
        -226,
        -143,
        342,
        -52,
        160,
        143,
        -108,
        -140,
        121,
        194,
        -446,
        -1,
        137,
        139,
        -61,
        283,
        -155,
        -106,
        -63,
        -32,
        -187,
        -23,
        206,
        -45,
        102,
        -201,
        -379,
        15,
        12,
        42,
        142,
        164,
        -101,
        -140,
        -213,
        -117,
        -27,
        -5,
        -184,
        -48,
        -137,
        -149,
        9,
        -138,
        58,
        -92,
        141,
        213,
        -46,
        153,
        43,
        84,
        -44,
        -61,
        168,
        91,
        -200,
        -101,
        -38,
        204,
        90,
        -291,
        -200,
        317,
        -252,
        54,
        270,
        224,
        79,
        122,
        -243,
        362,
        381,
        92,
        113,
        24,
        232,
        227,
        -262,
        -145,
        2,
        -195,
        -76,
        -142,
        37,
        129,
        8,
        -34,
        232,
        -15,
        68,
        -3,
        -100,
        184,
        -73,
        -94,
        -146,
        33,
        -78,
        -281,
        234,
        -151,
        -105,
        -269,
        221,
        -145,
        88,
        -435,
        -45,
        -153,
        -167,
        -189,
        -52,
        144,
        100,
        48,
        -108,
        -28,
        -215,
        -94,
        -131,
        302,
        132,
        21,
        -80,
        152,
        -85,
        -353,
        167,
    };

    test = Zf(neon_is_short_half1)(sqn, s2, 9);
    gold = ZfN(is_short_half)(sqn, s2, 9);

    printf("test, gold: %d -- %d\n", test, gold);
    if (test != gold)
    {
        return 1;
    }

    return 0;
}
