#include <arm_neon.h>
#include <stdio.h>

// gcc -o test reduction_test.c -O3; ./test

#define FALCON_Q 12289
#define FALCON_QINV (-12287)

int16_t mul(int16_t a, int16_t b)
{
    int16_t c;
    c = ((int32_t)a * b) % FALCON_Q;
    return c;
}

/*
 * If `sl, sr` = 25, 26 then `v` = 5461
 * If `sl, sr` = 26, 27 then `v` = 10922
 * If `sl, sr` = 27, 28 then `v` = 21844
 * I select `v` = 5461
 */
uint16_t barrett_reduce(uint16_t a)
{
    uint16_t t;
    const int sr = 28;
    const uint16_t v = (1 << sr) / FALCON_Q + 1;

    t = ((uint32_t)v * a) >> sr;
    t *= FALCON_Q;
    return a - t;
}

uint16_t barrett_simd_13(uint16_t a, int k, int ROUNDING, int SHIFT_ROUNDING, int SHIFT_SIGNED)
{
    uint16x8_t t, z;
    z = vdupq_n_u16(a);

    if (ROUNDING)
    {
        t = vqrdmulhq_n_s16(z, k);
    }
    else
    {
        t = vqdmulhq_n_s16(z, k);
    }

    if (SHIFT_ROUNDING)
    {
        if (SHIFT_SIGNED)
        {
            t = vrshrq_n_s16(t, 13);
        }
        else
        {
            t = vrshrq_n_u16(t, 13);
        }
    }
    else
    {
        if (SHIFT_SIGNED)
        {
            t = vshrq_n_s16(t, 13);
        }
        else
        {
            t = vshrq_n_u16(t, 13);
        }
    }

    z = vmlaq_n_u16(z, t, -FALCON_Q);
    return z[0];
}

uint16_t barrett_simd_12(uint16_t a, int k, int ROUNDING, int SHIFT_ROUNDING, int SHIFT_SIGNED)
{
    uint16x8_t t, z;
    z = vdupq_n_u16(a);
    if (ROUNDING)
    {
        t = vqrdmulhq_n_s16(z, k);
    }
    else
    {
        t = vqdmulhq_n_s16(z, k);
    }

    if (SHIFT_ROUNDING)
    {
        if (SHIFT_SIGNED)
        {
            t = vrshrq_n_s16(t, 12);
        }
        else
        {
            t = vrshrq_n_u16(t, 12);
        }
    }
    else
    {
        if (SHIFT_SIGNED)
        {
            t = vshrq_n_s16(t, 12);
        }
        else
        {
            t = vshrq_n_u16(t, 12);
        }
    }

    z = vmlaq_n_u16(z, t, -FALCON_Q);
    return z[0];
}

uint16_t barrett_simd_11(uint16_t a, int k, int ROUNDING, int SHIFT_ROUNDING, int SHIFT_SIGNED)
{
    uint16x8_t t, z;

    z = vdupq_n_u16(a);

    // z = vabsq_s16(z);

    if (ROUNDING)
    {
        t = vqrdmulhq_n_s16(z, k);
    }
    else
    {
        t = vqdmulhq_n_s16(z, k);
    }

    if (SHIFT_ROUNDING)
    {
        if (SHIFT_SIGNED)
        {
            t = vrshrq_n_s16(t, 11);
        }
        else
        {
            t = vrshrq_n_u16(t, 11);
        }
    }
    else
    {
        if (SHIFT_SIGNED)
        {
            t = vshrq_n_s16(t, 11);
        }
        else
        {
            t = vshrq_n_u16(t, 11);
        }
    }
    // printf("%d: t >>%d = %d - |%d|\n", a, 11, t[0] & 0xffff, a / FALCON_Q);

    z = vmlaq_n_s16(z, t, -FALCON_Q);
    return z[0];
}

uint16_t barrett_simd_14(uint16_t a, int k, int ROUNDING, int SHIFT_ROUNDING, int SHIFT_SIGNED)
{
    uint16x8_t t, z, bk;
    z = vdupq_n_u16(a);
    if (ROUNDING)
    {
        t = vqrdmulhq_n_s16(z, k);
    }
    else
    {
        t = vqdmulhq_n_s16(z, k);
    }

    bk = t;
    t = vsubq_u16(t, vdupq_n_u16(0));

    if (SHIFT_ROUNDING)
    {
        if (SHIFT_SIGNED)
        {
            t = vrshrq_n_s16(t, 14);
        }
        else
        {
            t = vrshrq_n_u16(t, 14);
        }
    }
    else
    {
        if (SHIFT_SIGNED)
        {
            t = vshrq_n_s16(t, 14);
        }
        else
        {
            t = vshrq_n_u16(t, 14);
        }
    }
    if (t[0] != a/FALCON_Q)
    {
        // printf("%d: %d, t >>%d = %d - |%d|\n", a, bk[0], 14, t[0] & 0xffff, a / FALCON_Q);
    }
    z = vmlaq_n_u16(z, t, -FALCON_Q);
    return z[0];
}

uint16_t barrett_simd_15(uint16_t a, int k, int ROUNDING, int SHIFT_ROUNDING, int SHIFT_SIGNED)
{
    uint16x8_t t, z;
    z = vdupq_n_u16(a);
    if (ROUNDING)
    {
        t = vqrdmulhq_n_s16(z, k);
    }
    else
    {
        t = vqdmulhq_n_s16(z, k);
    }

    // printf("t = %d: %d x %d\n", t[0] & 0xffff, a, k);

    if (SHIFT_ROUNDING)
    {
        if (SHIFT_SIGNED)
        {
            t = vrshrq_n_s16(t, 15);
        }
        else
        {
            t = vrshrq_n_u16(t, 15);
        }

        // printf("t = %d: %d x %d\n", t[0] & 0xffff, a, k);
    }
    else
    {
        if (SHIFT_SIGNED)
        {
            t = vshrq_n_s16(t, 15);
        }
        else
        {
            t = vshrq_n_u16(t, 15);
        }
    }
    // printf("%d: t >>%d = %d - |%d|\n", a, 11, t[0] & 0xffff, a / FALCON_Q);
    z = vmlaq_n_u16(z, t, -FALCON_Q);
    return z[0];
}

int search(uint16_t k, uint16_t (*func)(uint16_t, int, int, int, int), const char *string, int ROUNDING, int SHIFT_ROUNDING, int SHIFT_SIGNED, int verbose)
{
    if (verbose)
        printf("test_barrett_reduction: %s\n", string);

    uint16_t gold, test;
    uint16_t min = UINT16_MAX, max = 0;
    int count = 0;
    uint16_t start, end;
    unsigned already_set = 0;
    unsigned already_print = 0;

    for (uint16_t i = 0; i < 5*FALCON_Q; i++)
    {
        gold = i % FALCON_Q;
        test = func(i, k, ROUNDING, SHIFT_ROUNDING, SHIFT_SIGNED);

        if ((gold == test) && (test < FALCON_Q))
        {
            // printf("%u : %u == %u\n\n", i & 0xffff, gold &0xffff, test & 0xffff);
            count++;
            if (!already_set)
            {
                start = i;
                already_set = 1;
                already_print = 0;
            }
        }
        else
        {
            // printf("%u : %u != %u |%u|\n\n", i & 0xffff, gold &0xffff, test & 0xffff, (gold - test) & 0xffff);
            end = i;
            if (!already_print)
            {
                if (verbose)
                {
                    printf("%d: %d -> %d | %d | \n", i, start, end, end - start);
                }
                already_print = 1;
            }
            already_set = 0;
        }
    }
    if (max < count)
    {
        max = count;
    }

    if (verbose)
        printf("k = %d, count = %d\n", k, max);
    return count;
}