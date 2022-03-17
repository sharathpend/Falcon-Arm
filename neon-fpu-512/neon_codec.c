#include <arm_neon.h>
#include "config.h"
#include "ntt.h"

/*
 * Check if (t < low || t > high)
 * Return 1 if True
 * Otherwise 0
 */
int neon_bound_check_int8_low_high(const int8_t t[FALCON_N], 
                        const int8_t low, const int8_t high)
{
    // Total SIMD registers: 10
    int8x16x4_t a;                 // 4
    uint8x16x4_t c;                // 4
    uint8x16_t e;                  // 1
    int8x16_t neon_low, neon_high; // 1

    neon_high = vdupq_n_s8(high);
    neon_low = vdupq_n_s8(low);
    e = vdupq_n_u8(0);

    for (int i = 0; i < FALCON_N; i += 64)
    {
        a = vld1q_s8_x4(&t[i]);

        // low > a ? 1 : 0
        c.val[0] = vcgtq_s8(neon_low, a.val[0]);
        c.val[1] = vcgtq_s8(neon_low, a.val[1]);
        c.val[2] = vcgtq_s8(neon_low, a.val[2]);
        c.val[3] = vcgtq_s8(neon_low, a.val[3]);
        // a > high ? 1 : 0
        c.val[0] = vcgtq_s8(a.val[0], neon_high);
        c.val[1] = vcgtq_s8(a.val[1], neon_high);
        c.val[2] = vcgtq_s8(a.val[2], neon_high);
        c.val[3] = vcgtq_s8(a.val[3], neon_high);

        c.val[0] = vorrq_u8(c.val[0], c.val[0]);
        c.val[1] = vorrq_u8(c.val[1], c.val[1]);
        c.val[2] = vorrq_u8(c.val[2], c.val[2]);
        c.val[3] = vorrq_u8(c.val[3], c.val[3]);

        c.val[0] = vorrq_u8(c.val[0], c.val[2]);
        c.val[1] = vorrq_u8(c.val[1], c.val[3]);

        c.val[0] = vorrq_u8(c.val[0], c.val[1]);

        e = vorrq_u8(e, c.val[0]);

        if (vmaxvq_u8(e))
        {
            return 1;
        }
    }
    return 0;
}

/*
 * Check if (t < low || t > high)
 * Return 1 if True
 * Otherwise 0
 */
int neon_bound_check_int16_low_high(const int16_t t[FALCON_N], 
                    const int16_t low, const int16_t high)
{
    // Total SIMD registers: 10
    int16x8x4_t a;                 // 4
    uint16x8x4_t c;                // 4
    uint16x8_t e;                  // 1
    int16x8_t neon_low, neon_high; // 1

    neon_high = vdupq_n_s16(high);
    neon_low = vdupq_n_s16(low);
    e = vdupq_n_u16(0);

    for (int i = 0; i < FALCON_N; i += 32)
    {
        a = vld1q_s16_x4(&t[i]);

        // low > a ? 1 : 0
        c.val[0] = vcgtq_s16(neon_low, a.val[0]);
        c.val[1] = vcgtq_s16(neon_low, a.val[1]);
        c.val[2] = vcgtq_s16(neon_low, a.val[2]);
        c.val[3] = vcgtq_s16(neon_low, a.val[3]);
        // a > high ? 1 : 0
        c.val[0] = vcgtq_s16(a.val[0], neon_high);
        c.val[1] = vcgtq_s16(a.val[1], neon_high);
        c.val[2] = vcgtq_s16(a.val[2], neon_high);
        c.val[3] = vcgtq_s16(a.val[3], neon_high);

        c.val[0] = vorrq_u16(c.val[0], c.val[0]);
        c.val[1] = vorrq_u16(c.val[1], c.val[1]);
        c.val[2] = vorrq_u16(c.val[2], c.val[2]);
        c.val[3] = vorrq_u16(c.val[3], c.val[3]);

        c.val[0] = vorrq_u16(c.val[0], c.val[2]);
        c.val[1] = vorrq_u16(c.val[1], c.val[3]);

        c.val[0] = vorrq_u16(c.val[0], c.val[1]);

        e = vorrq_u16(e, c.val[0]);

        if (vmaxvq_u16(e))
        {
            return 1;
        }
    }
    return 0;
}