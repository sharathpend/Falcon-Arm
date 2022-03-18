#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include "../config.h"

int compare_array(int16_t *a, int16_t *b, int bound)
{
    for (int i = 0; i < bound; i++)
    {
        if ( (a[i] != b[i]) && (a[i] != b[i] + FALCON_Q) )
        {
            printf("%d: %d != %d\n", i, a[i], b[i]);
            return 1;
        }
    }
    return 0;
}

static inline uint32_t mq_conv_small(int x)
{
    uint32_t y;

    y = (uint32_t)x;
    y += FALCON_Q & -(y >> 31);
    return y;
}

void ZfN(poly_smallints_to_bigints)(int16_t out[FALCON_N], int8_t in[FALCON_N])
{
    int16x8x4_t a, b, e, f;
    int8x16x4_t c, d;
    
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


int test_conv_small(const int tests)
{
    int16_t out_gold[FALCON_N], out[FALCON_N];
    int8_t in[FALCON_N];
    int8_t temp;

    for (int i = 0; i < FALCON_N; i++)
    {
        temp = -i;
        in[i] = temp;

        out_gold[i] = (uint16_t)mq_conv_small(temp);
    }

    ZfN(poly_smallints_to_bigints)(out, in);

    if (compare_array(out_gold, out, FALCON_N))
    {
        printf("ERROR\n");
        return 1;
    }
    return 0;
}

#define TESTS 1

int main()
{
    int ret = 0;

    ret |= test_conv_small(TESTS);


    return ret;
}