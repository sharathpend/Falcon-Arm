#include <arm_neon.h>
#include "../inner.h"
#include "../macrous.h"
#include "../config.h"
#include <stdio.h>
#include <assert.h>

#define DEBUG 0

static const uint32_t l2bound[] = {
	0,    /* unused */
	101498,
	208714,
	428865,
	892039,
	1852696,
	3842630,
	7959734,
	16468416,
	34034726,
	70265242
};

/* see inner.h */
int
Zf(is_short)(
	const int16_t *s1, const int16_t *s2, unsigned logn)
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
	for (u = 0; u < n; u ++) {
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
    // // 32x4
    neon_s = vhaddq_s32(neon_s, neon_sh);
    // 32x4 -> 32x1
    s = vaddvq_s32(neon_s);

    printf("neon %8x\n", s);

    return s <= l2bound[FALCON_LOGN];
}

#define TESTS 100

void center_q(int16_t *a)
{
    for (int i = 0; i < FALCON_N; i++)
    {
        if (a[i] > FALCON_Q/2)
        {
            a[i] -= FALCON_Q;
        }
        else if (a[i] < -FALCON_Q/2)
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
