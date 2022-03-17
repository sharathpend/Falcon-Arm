/*
 * Support functions for signatures (hash-to-point, norm).
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2017-2019  Falcon Project
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   Thomas Pornin <thomas.pornin@nccgroup.com>
 */

#include "inner.h"
#include "macrous.h"
#include "macrofx4.h"

/* see inner.h */
void
    Zf(hash_to_point_vartime)(
        inner_shake256_context *sc,
        uint16_t *x, unsigned logn)
{
    /*
	 * This is the straightforward per-the-spec implementation. It
	 * is not constant-time, thus it might reveal information on the
	 * plaintext (at least, enough to check the plaintext against a
	 * list of potential plaintexts) in a scenario where the
	 * attacker does not have access to the signature value or to
	 * the public key, but knows the nonce (without knowledge of the
	 * nonce, the hashed output cannot be matched against potential
	 * plaintexts).
	 */
    size_t n;

    n = (size_t)1 << logn;
    while (n > 0)
    {
        uint8_t buf[2];
        uint32_t w;

        inner_shake256_extract(sc, (void *)buf, sizeof buf);
        w = ((unsigned)buf[0] << 8) | (unsigned)buf[1];
        if (w < 61445)
        {
            while (w >= 12289)
            {
                w -= 12289;
            }
            *x++ = (uint16_t)w;
            n--;
        }
    }
}

/* see inner.h */
void
    Zf(hash_to_point_ct)(
        inner_shake256_context *sc,
        uint16_t *x, unsigned logn, uint8_t *tmp)
{
    /*
	 * Each 16-bit sample is a value in 0..65535. The value is
	 * kept if it falls in 0..61444 (because 61445 = 5*12289)
	 * and rejected otherwise; thus, each sample has probability
	 * about 0.93758 of being selected.
	 *
	 * We want to oversample enough to be sure that we will
	 * have enough values with probability at least 1 - 2^(-256).
	 * Depending on degree N, this leads to the following
	 * required oversampling:
	 *
	 *   logn     n  oversampling
	 *     1      2     65
	 *     2      4     67
	 *     3      8     71
	 *     4     16     77
	 *     5     32     86
	 *     6     64    100
	 *     7    128    122
	 *     8    256    154
	 *     9    512    205
	 *    10   1024    287
	 *
	 * If logn >= 7, then the provided temporary buffer is large
	 * enough. Otherwise, we use a stack buffer of 63 entries
	 * (i.e. 126 bytes) for the values that do not fit in tmp[].
	 */

    static const uint16_t overtab[] = {
        0, /* unused */
        65,
        67,
        71,
        77,
        86,
        100,
        122,
        154,
        205,
        287};

    unsigned n, n2, u, m, p, over;
    uint16_t *tt1, tt2[63];

    /*
	 * We first generate m 16-bit value. Values 0..n-1 go to x[].
	 * Values n..2*n-1 go to tt1[]. Values 2*n and later go to tt2[].
	 * We also reduce modulo q the values; rejected values are set
	 * to 0xFFFF.
	 */
    n = 1U << logn;
    n2 = n << 1;
    over = overtab[logn];
    m = n + over;
    tt1 = (uint16_t *)tmp;
    for (u = 0; u < m; u++)
    {
        uint8_t buf[2];
        uint32_t w, wr;

        inner_shake256_extract(sc, buf, sizeof buf);
        w = ((uint32_t)buf[0] << 8) | (uint32_t)buf[1];
        wr = w - ((uint32_t)24578 & (((w - 24578) >> 31) - 1));
        wr = wr - ((uint32_t)24578 & (((wr - 24578) >> 31) - 1));
        wr = wr - ((uint32_t)12289 & (((wr - 12289) >> 31) - 1));
        wr |= ((w - 61445) >> 31) - 1;
        if (u < n)
        {
            x[u] = (uint16_t)wr;
        }
        else if (u < n2)
        {
            tt1[u - n] = (uint16_t)wr;
        }
        else
        {
            tt2[u - n2] = (uint16_t)wr;
        }
    }

    /*
	 * Now we must "squeeze out" the invalid values. We do this in
	 * a logarithmic sequence of passes; each pass computes where a
	 * value should go, and moves it down by 'p' slots if necessary,
	 * where 'p' uses an increasing powers-of-two scale. It can be
	 * shown that in all cases where the loop decides that a value
	 * has to be moved down by p slots, the destination slot is
	 * "free" (i.e. contains an invalid value).
	 */
    for (p = 1; p <= over; p <<= 1)
    {
        unsigned v;

        /*
		 * In the loop below:
		 *
		 *   - v contains the index of the final destination of
		 *     the value; it is recomputed dynamically based on
		 *     whether values are valid or not.
		 *
		 *   - u is the index of the value we consider ("source");
		 *     its address is s.
		 *
		 *   - The loop may swap the value with the one at index
		 *     u-p. The address of the swap destination is d.
		 */
        v = 0;
        for (u = 0; u < m; u++)
        {
            uint16_t *s, *d;
            unsigned j, sv, dv, mk;

            if (u < n)
            {
                s = &x[u];
            }
            else if (u < n2)
            {
                s = &tt1[u - n];
            }
            else
            {
                s = &tt2[u - n2];
            }
            sv = *s;

            /*
			 * The value in sv should ultimately go to
			 * address v, i.e. jump back by u-v slots.
			 */
            j = u - v;

            /*
			 * We increment v for the next iteration, but
			 * only if the source value is valid. The mask
			 * 'mk' is -1 if the value is valid, 0 otherwise,
			 * so we _subtract_ mk.
			 */
            mk = (sv >> 15) - 1U;
            v -= mk;

            /*
			 * In this loop we consider jumps by p slots; if
			 * u < p then there is nothing more to do.
			 */
            if (u < p)
            {
                continue;
            }

            /*
			 * Destination for the swap: value at address u-p.
			 */
            if ((u - p) < n)
            {
                d = &x[u - p];
            }
            else if ((u - p) < n2)
            {
                d = &tt1[(u - p) - n];
            }
            else
            {
                d = &tt2[(u - p) - n2];
            }
            dv = *d;

            /*
			 * The swap should be performed only if the source
			 * is valid AND the jump j has its 'p' bit set.
			 */
            mk &= -(((j & p) + 0x1FF) >> 9);

            *s = (uint16_t)(sv ^ (mk & (sv ^ dv)));
            *d = (uint16_t)(dv ^ (mk & (sv ^ dv)));
        }
    }
}

/*
 * Acceptance bound for the (squared) l2-norm of the signature depends
 * on the degree. This array is indexed by logn (1 to 10). These bounds
 * are _inclusive_ (they are equal to floor(beta^2)).
 */
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
int Zf(is_short)(const int16_t *s1, const int16_t *s2)
{
    int16x8x4_t neon_s1, neon_s2;
    uint32x4_t neon_ng, neon_ngh;
    int32x4_t neon_s, neon_sh, neon_zero;
    uint32_t s, sh, ng;
    neon_s = vdupq_n_s32(0);
    neon_sh = vdupq_n_s32(0);
    neon_zero = vdupq_n_s32(0);
    neon_ng = vdupq_n_u32(0);
    neon_ngh = vdupq_n_u32(0);

    for (unsigned u = 0; u < FALCON_N; u += 32)
    {
        neon_s1 = vld1q_s16_x4(&s1[u]);
        neon_s2 = vld1q_s16_x4(&s2[u]);

        vmulla_lo(neon_s, neon_s, neon_s1.val[0], neon_s1.val[0]);
        vmulla_hi(neon_sh, neon_sh, neon_s1.val[0], neon_s1.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);

        vmulla_lo(neon_s, neon_s, neon_s1.val[1], neon_s1.val[1]);
        vmulla_hi(neon_sh, neon_sh, neon_s1.val[1], neon_s1.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);

        vmulla_lo(neon_s, neon_s, neon_s1.val[2], neon_s1.val[2]);
        vmulla_hi(neon_sh, neon_sh, neon_s1.val[2], neon_s1.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);

        vmulla_lo(neon_s, neon_s, neon_s1.val[3], neon_s1.val[3]);
        vmulla_hi(neon_sh, neon_sh, neon_s1.val[3], neon_s1.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);
        //
        vmulla_lo(neon_s, neon_s, neon_s2.val[0], neon_s2.val[0]);
        vmulla_hi(neon_sh, neon_sh, neon_s2.val[0], neon_s2.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);

        vmulla_lo(neon_s, neon_s, neon_s2.val[1], neon_s2.val[1]);
        vmulla_hi(neon_sh, neon_sh, neon_s2.val[1], neon_s2.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);

        vmulla_lo(neon_s, neon_s, neon_s2.val[2], neon_s2.val[2]);
        vmulla_hi(neon_sh, neon_sh, neon_s2.val[2], neon_s2.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);

        vmulla_lo(neon_s, neon_s, neon_s2.val[3], neon_s2.val[3]);
        vmulla_hi(neon_sh, neon_sh, neon_s2.val[3], neon_s2.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);
    }
    // 32x2
    neon_s = vpaddq_s32(neon_s, neon_zero);
    neon_sh = vpaddq_s32(neon_sh, neon_zero);
    vor(neon_ng, neon_ng, (uint32x4_t)neon_s);
    vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sh);
    vor(neon_ng, neon_ng, neon_ngh);
    s = vaddvq_s32(neon_s);
    sh = vaddvq_s32(neon_sh);

    ng = vgetq_lane_u32(neon_ng, 0);
    ng |= vgetq_lane_u32(neon_ng, 1);
    ng |= vgetq_lane_u32(neon_ng, 2);
    ng |= vgetq_lane_u32(neon_ng, 3);
    ng |= s;
    ng |= sh;

    // printf("s: %8x\n", s);
    s |= -(ng >> 31);
    // printf("is_short s, ng: %8x | %8x\n", s, ng);

    return s <= l2bound[FALCON_LOGN];
}

/* see inner.h */
int Zf(is_short_half)(uint32_t sqn, const int16_t *s2)
{
    int16x8x4_t s2_s16;
    int32x4_t neon_sqn, neon_sqnh, neon_zero;
    uint32x4_t neon_ng, neon_ngh;
    uint32_t ng = -(sqn >> 31);

    neon_sqn = vdupq_n_s32(0);
    neon_sqnh = vdupq_n_s32(0);
    neon_zero = vdupq_n_s32(0);
    neon_ng = vdupq_n_u32(0);
    neon_ngh = vdupq_n_u32(0);

    for (unsigned u = 0; u < FALCON_N; u += 32)
    {
        s2_s16 = vld1q_s16_x4(&s2[u]);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[0], s2_s16.val[0]);
        vmulla_hi(neon_sqnh, neon_sqnh, s2_s16.val[0], s2_s16.val[0]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sqnh);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[1], s2_s16.val[1]);
        vmulla_hi(neon_sqnh, neon_sqnh, s2_s16.val[1], s2_s16.val[1]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sqnh);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[2], s2_s16.val[2]);
        vmulla_hi(neon_sqnh, neon_sqnh, s2_s16.val[2], s2_s16.val[2]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sqnh);

        vmulla_lo(neon_sqn, neon_sqn, s2_s16.val[3], s2_s16.val[3]);
        vmulla_hi(neon_sqnh, neon_sqnh, s2_s16.val[3], s2_s16.val[3]);
        vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
        vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sqnh);
    }
    // 32x2
    neon_sqn = vpaddq_s32(neon_sqn, neon_zero);
    neon_sqnh = vpaddq_s32(neon_sqnh, neon_zero);
    vor(neon_ng, neon_ng, (uint32x4_t)neon_sqn);
    vor(neon_ngh, neon_ngh, (uint32x4_t)neon_sqnh);
    vor(neon_ng, neon_ng, neon_ngh);
    // ng |= sqn;
    sqn += vaddvq_s32(neon_sqn);
    ng |= sqn;
    sqn += vaddvq_s32(neon_sqnh);
    ng |= sqn;
    ng |= vgetq_lane_u32(neon_ng, 0);
    ng |= vgetq_lane_u32(neon_ng, 1);
    ng |= vgetq_lane_u32(neon_ng, 2);
    ng |= vgetq_lane_u32(neon_ng, 3);

    // printf("is_short_half sqn: %8x\n", sqn);

    sqn |= -(ng >> 31);

    // printf("is_short_half ng, sqn: %8x | %8x\n", ng, sqn);

    return sqn <= l2bound[FALCON_LOGN];
}

void Zf(sign_short_s1)(uint32_t *sqn_out, int16_t *s1tmp, const uint16_t *hm, const double *t0, const unsigned falcon_n)
{
    float64x2x4_t neon_tf64[2];
    int64x2x4_t neon_ts64[2];
    int32x4x4_t neon_ts32, neon_hms32, z;
    uint16x8x2_t neon_hm;
    int16x8x2_t z16;
    uint32x4_t neon_sqn, neon_ng;
    uint16x8_t neon_zero;
    uint32_t ng = 0, sqn = 0;

    neon_sqn = vdupq_n_u32(0);
    neon_ng = vdupq_n_u32(0);
    neon_zero = vdupq_n_u16(0);
    

    for (unsigned u = 0; u < falcon_n; u += 16)
    {

        vloadx4(neon_tf64[0], &t0[u]);
        vloadx4(neon_tf64[1], &t0[u + 8]);
        neon_hm = vld1q_u16_x2(&hm[u]);

        vfrintx4(neon_ts64[0], neon_tf64[0]);
        vfrintx4(neon_ts64[1], neon_tf64[1]);

        neon_ts32.val[0] = vuzp1q_s32( (int32x4_t) neon_ts64[0].val[0], (int32x4_t) neon_ts64[0].val[1]);
        neon_ts32.val[1] = vuzp1q_s32( (int32x4_t) neon_ts64[0].val[2], (int32x4_t) neon_ts64[0].val[3]);
        neon_ts32.val[2] = vuzp1q_s32( (int32x4_t) neon_ts64[1].val[0], (int32x4_t) neon_ts64[1].val[1]);
        neon_ts32.val[3] = vuzp1q_s32( (int32x4_t) neon_ts64[1].val[2], (int32x4_t) neon_ts64[1].val[3]);

        neon_hms32.val[0] = (int32x4_t)vzip1q_u16( (uint16x8_t) neon_hm.val[0], (uint16x8_t) neon_zero);
        neon_hms32.val[1] = (int32x4_t)vzip2q_u16( (uint16x8_t) neon_hm.val[0], (uint16x8_t) neon_zero);
        neon_hms32.val[2] = (int32x4_t)vzip1q_u16( (uint16x8_t) neon_hm.val[1], (uint16x8_t) neon_zero);
        neon_hms32.val[3] = (int32x4_t)vzip2q_u16( (uint16x8_t) neon_hm.val[1], (uint16x8_t) neon_zero);

        z.val[0] = vsubq_s32(neon_hms32.val[0], neon_ts32.val[0]);
        z.val[1] = vsubq_s32(neon_hms32.val[1], neon_ts32.val[1]);
        z.val[2] = vsubq_s32(neon_hms32.val[2], neon_ts32.val[2]);
        z.val[3] = vsubq_s32(neon_hms32.val[3], neon_ts32.val[3]);

        neon_sqn = vmlaq_u32(neon_sqn, (uint32x4_t)z.val[0], (uint32x4_t)z.val[0]);
        neon_ng = vorrq_u32(neon_ng, neon_sqn);

        neon_sqn = vmlaq_u32(neon_sqn, (uint32x4_t)z.val[1], (uint32x4_t)z.val[1]);
        neon_ng = vorrq_u32(neon_ng, neon_sqn);

        neon_sqn = vmlaq_u32(neon_sqn, (uint32x4_t)z.val[2], (uint32x4_t)z.val[2]);
        neon_ng = vorrq_u32(neon_ng, neon_sqn);

        neon_sqn = vmlaq_u32(neon_sqn, (uint32x4_t)z.val[3], (uint32x4_t)z.val[3]);
        neon_ng = vorrq_u32(neon_ng, neon_sqn);

        z16.val[0] = vuzp1q_s16( (int16x8_t) z.val[0], (int16x8_t) z.val[1]);
        z16.val[1] = vuzp1q_s16( (int16x8_t) z.val[2], (int16x8_t) z.val[3]);

        vst1q_s16_x2(&s1tmp[u], z16);
    }
    sqn += vgetq_lane_u32(neon_sqn, 0);
    ng |= sqn;
    sqn += vgetq_lane_u32(neon_sqn, 1);
    ng |= sqn;
    sqn += vgetq_lane_u32(neon_sqn, 2);
    ng |= sqn;
    sqn += vgetq_lane_u32(neon_sqn, 3);
    ng |= sqn;
    ng |= vgetq_lane_u32(neon_ng, 0);
    ng |= vgetq_lane_u32(neon_ng, 1);
    ng |= vgetq_lane_u32(neon_ng, 2);
    ng |= vgetq_lane_u32(neon_ng, 3);

    // printf("sqn %u\n", sqn);

    sqn |= -(ng >> 31);

    *sqn_out = sqn;
}

void Zf(sign_short_s2)(int16_t *s2tmp, const double *t1, const unsigned falcon_n)
{
    float64x2x4_t neon_tf64[4];
    int64x2x4_t neon_ts64[4];
    int32x4x4_t neon_ts32[2];
    int16x8x4_t neon_s2;
    for (unsigned u = 0; u < falcon_n; u += 32)
    {
        vloadx4(neon_tf64[0], &t1[u]);
        vloadx4(neon_tf64[1], &t1[u + 8]);
        vloadx4(neon_tf64[2], &t1[u + 16]);
        vloadx4(neon_tf64[3], &t1[u + 24]);

        vfrintx4(neon_ts64[0], neon_tf64[0]);
        vfrintx4(neon_ts64[1], neon_tf64[1]);
        vfrintx4(neon_ts64[2], neon_tf64[2]);
        vfrintx4(neon_ts64[3], neon_tf64[3]);

        neon_ts32[0].val[0] = vuzp1q_s32((int32x4_t) neon_ts64[0].val[0], (int32x4_t) neon_ts64[0].val[1]);
        neon_ts32[0].val[1] = vuzp1q_s32((int32x4_t) neon_ts64[0].val[2], (int32x4_t) neon_ts64[0].val[3]);
        neon_ts32[0].val[2] = vuzp1q_s32((int32x4_t) neon_ts64[1].val[0], (int32x4_t) neon_ts64[1].val[1]);
        neon_ts32[0].val[3] = vuzp1q_s32((int32x4_t) neon_ts64[1].val[2], (int32x4_t) neon_ts64[1].val[3]);

        neon_ts32[1].val[0] = vuzp1q_s32((int32x4_t) neon_ts64[2].val[0], (int32x4_t) neon_ts64[2].val[1]);
        neon_ts32[1].val[1] = vuzp1q_s32((int32x4_t) neon_ts64[2].val[2], (int32x4_t) neon_ts64[2].val[3]);
        neon_ts32[1].val[2] = vuzp1q_s32((int32x4_t) neon_ts64[3].val[0], (int32x4_t) neon_ts64[3].val[1]);
        neon_ts32[1].val[3] = vuzp1q_s32((int32x4_t) neon_ts64[3].val[2], (int32x4_t) neon_ts64[3].val[3]);

        neon_s2.val[0] = vuzp1q_s16((int16x8_t) neon_ts32[0].val[0], (int16x8_t) neon_ts32[0].val[1]);
        neon_s2.val[1] = vuzp1q_s16((int16x8_t) neon_ts32[0].val[2], (int16x8_t) neon_ts32[0].val[3]);
        neon_s2.val[2] = vuzp1q_s16((int16x8_t) neon_ts32[1].val[0], (int16x8_t) neon_ts32[1].val[1]);
        neon_s2.val[3] = vuzp1q_s16((int16x8_t) neon_ts32[1].val[2], (int16x8_t) neon_ts32[1].val[3]);

        neon_s2.val[0] = vnegq_s16(neon_s2.val[0]);
        neon_s2.val[1] = vnegq_s16(neon_s2.val[1]);
        neon_s2.val[2] = vnegq_s16(neon_s2.val[2]);
        neon_s2.val[3] = vnegq_s16(neon_s2.val[3]);

        vst1q_s16_x4(&s2tmp[u], neon_s2);
    }
}
