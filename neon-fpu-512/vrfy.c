/*
 * Falcon signature verification.
 *
 * =============================================================================
 * Copyright (c) 2021 by Cryptographic Engineering Research Group (CERG)
 * ECE Department, George Mason University
 * Fairfax, VA, U.S.A.
 * Author: Duc Tri Nguyen
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 * @author   Duc Tri Nguyen <dnguye69@gmu.edu>
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

    neon_conv_small(h, g);
    neon_fwdNTT(h, 0);
   
    neon_conv_small(tt, f);
    neon_fwdNTT(tt, 1);

    if (neon_compare_with_zero(tt))
    {
        return 0;
    }
    neon_div_12289(h, tt);

    neon_invNTT(h);

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
    neon_fwdNTT(t1, 0);

    neon_conv_small(t2, F);
    neon_fwdNTT(t2, 1);

    neon_poly_montymul_ntt(t1, t2);

    neon_conv_small(t2, f);
    neon_fwdNTT(t2, 1);

    if (neon_compare_with_zero(t2))
    {
        return 0;
    }
    neon_div_12289(t1, t2);

    neon_invNTT(t1);

    if (neon_big_to_smallints(G, t1))
    {
        return 0;
    }
    return 1;
}

/* see inner.h */
int Zf(is_invertible)(const int16_t *s2, uint8_t *tmp)
{
    int16_t *tt = (int16_t *)tmp;
    uint16_t r;

    memcpy(tt, s2, sizeof(int16_t) * FALCON_N);
    neon_fwdNTT(tt, 1);

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
     * Compute h = (c0 - s1) / s2. If one of the coefficients of s2
     * is zero (in NTT representation) then the operation fails. We
     * keep that information into a flag so that we do not deviate
     * from strict constant-time processing; if all coefficients of
     * s2 are non-zero, then the high bit of r will be zero.
     */

    neon_poly_sub(h, c0, s1);
    neon_fwdNTT(h, 0);

    /*
     * Reduce elements of s1 and s2 modulo q; then write s2 into tt[]
     * and c0 - s1 into h[].
     */
    memcpy(tt, s2, sizeof(int16_t) * FALCON_N);
    neon_fwdNTT(tt, 1);
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
    neon_fwdNTT(s2, 1);

    int r = neon_compare_with_zero(s2);

    return r;
}
