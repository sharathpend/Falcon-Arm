/*
 * Falcon signature verification.
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
#include "ntt.h"

/* see inner.h */
void Zf(to_ntt_monty)(int16_t *h)
{
    neon_fwdNTT(h, 1);
}

/* see inner.h */
int Zf(verify_raw)(const int16_t *c0, const int16_t *s2,
                   const int16_t *h, int16_t *tmp)
{
    int16_t *tt = tmp;

    /*
     * Compute -s1 = s2*h - c0 mod phi mod q (in tt[]).
     */

    memcpy(tt, s2, sizeof(int16_t) * FALCON_N);
    neon_fwdNTT(tt, 0);
    neon_poly_montymul_ntt(tt, h);
    neon_invNTT(tt);
    neon_poly_sub_barrett(tt, c0);

    /*
     * Signature is valid if and only if the aggregate (-s1,s2) vector
     * is short enough.
     */
    return Zf(is_short)(tt, s2);
}

/* see inner.h */
int Zf(compute_public)(int16_t *h, const int8_t *f, const int8_t *g, int16_t *tmp)
{
    int16_t *tt = tmp;

    neon_conv_small(tt, f);
    neon_fwdNTT(tt, 0);

    neon_conv_small(h, g);
    neon_fwdNTT(h, 0);

    if (neon_compare_with_zero(tt))
    {
        return 0;
    }
    neon_div_12289(h, tt);

    neon_invNTT(h);

    // TODO: add option to combine to invNTT
    neon_poly_unsigned(h);

    return 1;
}

/* see inner.h */
int Zf(complete_private)(int8_t *G, const int8_t *f,
                         const int8_t *g, const int8_t *F,
                         uint8_t *tmp)
{
    int16_t *t1, *t2;

    t1 = (int16_t *)tmp;
    t2 = t1 + FALCON_N;

    neon_conv_small(t1, g);
    neon_fwdNTT(t1, 1);
    neon_conv_small(t2, F);
    neon_fwdNTT(t2, 0);

    neon_poly_montymul_ntt(t1, t2);

    neon_conv_small(t2, f);
    neon_fwdNTT(t2, 0);

    if (neon_compare_with_zero(t2))
    {
        return 0;
    }
    neon_div_12289(t1, t2);

    neon_invNTT(t1);

    for (size_t u = 0; u < FALCON_N; u++)
    {
        uint32_t w;
        int32_t gi;

        w = t1[u];
        w -= (Q & ~-((w - (Q >> 1)) >> 31));
        gi = *(int32_t *)&w;
        if (gi < -127 || gi > +127)
        {
            return 0;
        }
        G[u] = (int8_t)gi;
    }
    return 1;
}

/* see inner.h */
int Zf(is_invertible)(const int16_t *s2, uint8_t *tmp)
{
    int16_t *tt = (int16_t *)tmp;
    uint16_t r;

    memcpy(tt, s2, sizeof(int16_t) * FALCON_N);

    neon_fwdNTT(tt, 0);

    r = neon_compare_with_zero(tt);

    return (int)(1u - (r >> 15));
}

/* see inner.h */
int Zf(verify_recover)(int16_t *h, const int16_t *c0,
                       const int16_t *s1, const int16_t *s2,
                       uint8_t *tmp)
{
    int16_t *tt = (int16_t *)tmp;
    uint16_t r;

    /*
     * Reduce elements of s1 and s2 modulo q; then write s2 into tt[]
     * and c0 - s1 into h[].
     */
    memcpy(tt, s2, sizeof(int16_t) * FALCON_N);
    neon_fwdNTT(tt, 0);

    /*
     * Compute h = (c0 - s1) / s2. If one of the coefficients of s2
     * is zero (in NTT representation) then the operation fails. We
     * keep that information into a flag so that we do not deviate
     * from strict constant-time processing; if all coefficients of
     * s2 are non-zero, then the high bit of r will be zero.
     */

    neon_poly_sub(h, c0, s1);
    neon_fwdNTT(h, 0);

    r = neon_compare_with_zero(tt);
    neon_div_12289(h, tt);

    neon_invNTT(h);

    /*
     * Signature is acceptable if and only if it is short enough,
     * and s2 was invertible mod phi mod q. The caller must still
     * check that the rebuilt public key matches the expected
     * value (e.g. through a hash).
     */
    r = ~r & (uint16_t)-Zf(is_short)(s1, s2);
    return (int)(r >> 15);
}

/* see inner.h */
int Zf(count_nttzero)(const int16_t *sig, uint8_t *tmp)
{
    int16_t *s2 = (int16_t *)tmp;

    memcpy(s2, sig, sizeof(int16_t) * FALCON_N);
    neon_fwdNTT(s2, 0);

    int r = neon_compare_with_zero(s2);

    return r;
}
