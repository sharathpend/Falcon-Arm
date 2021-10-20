#include "sign.h"
#include "inner.h"
/*
 * Falcon signature generation.
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


/* =================================================================== */

/*
 * Perform Fast Fourier Sampling for target vector t. The Gram matrix
 * is provided (G = [[g00, g01], [adj(g01), g11]]). The sampled vector
 * is written over (t0,t1). The Gram matrix is modified as well. The
 * tmp[] buffer must have room for four polynomials.
 */
static void
ffSampling_fft_dyntree(samplerZ samp, void *samp_ctx,
                       fpr *t0, fpr *t1,
                       fpr *g00, fpr *g01, fpr *g11,
                       unsigned logn, fpr *tmp) {
    size_t n, hn;
    fpr *z0, *z1;

    /*
     * Deepest level: the LDL tree leaf value is just g00 (the
     * array has length only 1 at this point); we normalize it
     * with regards to sigma, then use it for sampling.
     */
    if (logn == 0) {
        fpr leaf;

        leaf = g00[0];
        leaf = fpr_mul(fpr_sqrt(leaf), fpr_inv_sigma);
        t0[0] = fpr_of(samp(samp_ctx, t0[0], leaf));
        t1[0] = fpr_of(samp(samp_ctx, t1[0], leaf));
        return;
    }

    n = (size_t)1 << logn;
    hn = n >> 1;

    /*
     * Decompose G into LDL. We only need d00 (identical to g00),
     * d11, and l10; we do that in place.
     */
    PQCLEAN_FALCON512_NEON_poly_LDL_fft(g00, g01, g11, logn);

    /*
     * Split d00 and d11 and expand them into half-size quasi-cyclic
     * Gram matrices. We also save l10 in tmp[].
     */
    PQCLEAN_FALCON512_NEON_poly_split_fft(tmp, tmp + hn, g00, logn);
    memcpy(g00, tmp, n * sizeof * tmp);
    PQCLEAN_FALCON512_NEON_poly_split_fft(tmp, tmp + hn, g11, logn);
    memcpy(g11, tmp, n * sizeof * tmp);
    memcpy(tmp, g01, n * sizeof * g01);
    memcpy(g01, g00, hn * sizeof * g00);
    memcpy(g01 + hn, g11, hn * sizeof * g00);

    /*
     * The half-size Gram matrices for the recursive LDL tree
     * building are now:
     *   - left sub-tree: g00, g00+hn, g01
     *   - right sub-tree: g11, g11+hn, g01+hn
     * l10 is in tmp[].
     */

    /*
     * We split t1 and use the first recursive call on the two
     * halves, using the right sub-tree. The result is merged
     * back into tmp + 2*n.
     */
    z1 = tmp + n;
    PQCLEAN_FALCON512_NEON_poly_split_fft(z1, z1 + hn, t1, logn);
    ffSampling_fft_dyntree(samp, samp_ctx, z1, z1 + hn,
                           g11, g11 + hn, g01 + hn, logn - 1, z1 + n);
    PQCLEAN_FALCON512_NEON_poly_merge_fft(tmp + (n << 1), z1, z1 + hn, logn);

    /*
     * Compute tb0 = t0 + (t1 - z1) * l10.
     * At that point, l10 is in tmp, t1 is unmodified, and z1 is
     * in tmp + (n << 1). The buffer in z1 is free.
     *
     * In the end, z1 is written over t1, and tb0 is in t0.
     */
    memcpy(z1, t1, n * sizeof * t1);
    PQCLEAN_FALCON512_NEON_poly_sub(z1, tmp + (n << 1), logn);
    memcpy(t1, tmp + (n << 1), n * sizeof * tmp);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(tmp, z1, logn);
    PQCLEAN_FALCON512_NEON_poly_add(t0, tmp, logn);

    /*
     * Second recursive invocation, on the split tb0 (currently in t0)
     * and the left sub-tree.
     */
    z0 = tmp;
    PQCLEAN_FALCON512_NEON_poly_split_fft(z0, z0 + hn, t0, logn);
    ffSampling_fft_dyntree(samp, samp_ctx, z0, z0 + hn,
                           g00, g00 + hn, g01, logn - 1, z0 + n);
    PQCLEAN_FALCON512_NEON_poly_merge_fft(t0, z0, z0 + hn, logn);
}




/*
 * Compute a signature: the signature contains two vectors, s1 and s2.
 * The s1 vector is not returned. The squared norm of (s1,s2) is
 * computed, and if it is short enough, then s2 is returned into the
 * s2[] buffer, and 1 is returned; otherwise, s2[] is untouched and 0 is
 * returned; the caller should then try again.
 *
 * tmp[] must have room for at least nine polynomials.
 */
static int
do_sign_dyn(samplerZ samp, void *samp_ctx, int16_t *s2,
            const int8_t *f, const int8_t *g,
            const int8_t *F, const int8_t *G,
            const uint16_t *hm, unsigned logn, fpr *tmp) {
    size_t n, u;
    fpr *t0, *t1, *tx, *ty;
    fpr *b00, *b01, *b10, *b11, *g00, *g01, *g11;
    fpr ni;
    uint32_t sqn, ng;
    int16_t *s1tmp, *s2tmp;

    n = MKN(logn);

    /*
     * Lattice basis is B = [[g, -f], [G, -F]]. We convert it to FFT.
     */
    b00 = tmp;
    b01 = b00 + n;
    b10 = b01 + n;
    b11 = b10 + n;
    smallints_to_fpr(b01, f, logn);
    smallints_to_fpr(b00, g, logn);
    smallints_to_fpr(b11, F, logn);
    smallints_to_fpr(b10, G, logn);
    PQCLEAN_FALCON512_NEON_FFT(b01, logn);
    PQCLEAN_FALCON512_NEON_FFT(b00, logn);
    PQCLEAN_FALCON512_NEON_FFT(b11, logn);
    PQCLEAN_FALCON512_NEON_FFT(b10, logn);
    PQCLEAN_FALCON512_NEON_poly_neg(b01, logn);
    PQCLEAN_FALCON512_NEON_poly_neg(b11, logn);

    /*
     * Compute the Gram matrix G = B x B*. Formulas are:
     *   g00 = b00*adj(b00) + b01*adj(b01)
     *   g01 = b00*adj(b10) + b01*adj(b11)
     *   g10 = b10*adj(b00) + b11*adj(b01)
     *   g11 = b10*adj(b10) + b11*adj(b11)
     *
     * For historical reasons, this implementation uses
     * g00, g01 and g11 (upper triangle). g10 is not kept
     * since it is equal to adj(g01).
     *
     * We _replace_ the matrix B with the Gram matrix, but we
     * must keep b01 and b11 for computing the target vector.
     */
    t0 = b11 + n;
    t1 = t0 + n;

    memcpy(t0, b01, n * sizeof * b01);
    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(t0, logn);    // t0 <- b01*adj(b01)

    memcpy(t1, b00, n * sizeof * b00);
    PQCLEAN_FALCON512_NEON_poly_muladj_fft(t1, b10, logn);   // t1 <- b00*adj(b10)
    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(b00, logn);   // b00 <- b00*adj(b00)
    PQCLEAN_FALCON512_NEON_poly_add(b00, t0, logn);      // b00 <- g00
    memcpy(t0, b01, n * sizeof * b01);
    PQCLEAN_FALCON512_NEON_poly_muladj_fft(b01, b11, logn);  // b01 <- b01*adj(b11)
    PQCLEAN_FALCON512_NEON_poly_add(b01, t1, logn);      // b01 <- g01

    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(b10, logn);   // b10 <- b10*adj(b10)
    memcpy(t1, b11, n * sizeof * b11);
    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(t1, logn);    // t1 <- b11*adj(b11)
    PQCLEAN_FALCON512_NEON_poly_add(b10, t1, logn);      // b10 <- g11

    /*
     * We rename variables to make things clearer. The three elements
     * of the Gram matrix uses the first 3*n slots of tmp[], followed
     * by b11 and b01 (in that order).
     */
    g00 = b00;
    g01 = b01;
    g11 = b10;
    b01 = t0;
    t0 = b01 + n;
    t1 = t0 + n;

    /*
     * Memory layout at that point:
     *   g00 g01 g11 b11 b01 t0 t1
     */

    /*
     * Set the target vector to [hm, 0] (hm is the hashed message).
     */
    for (u = 0; u < n; u ++) {
        t0[u] = fpr_of(hm[u]);
        /* This is implicit.
        t1[u] = fpr_zero;
        */
    }

    /*
     * Apply the lattice basis to obtain the real target
     * vector (after normalization with regards to modulus).
     */
    PQCLEAN_FALCON512_NEON_FFT(t0, logn);
    ni = fpr_inverse_of_q;
    memcpy(t1, t0, n * sizeof * t0);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(t1, b01, logn);
    PQCLEAN_FALCON512_NEON_poly_mulconst(t1, fpr_neg(ni), logn);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(t0, b11, logn);
    PQCLEAN_FALCON512_NEON_poly_mulconst(t0, ni, logn);

    /*
     * b01 and b11 can be discarded, so we move back (t0,t1).
     * Memory layout is now:
     *      g00 g01 g11 t0 t1
     */
    memcpy(b11, t0, n * 2 * sizeof * t0);
    t0 = g11 + n;
    t1 = t0 + n;

    /*
     * Apply sampling; result is written over (t0,t1).
     */
    ffSampling_fft_dyntree(samp, samp_ctx,
                           t0, t1, g00, g01, g11, logn, t1 + n);

    /*
     * We arrange the layout back to:
     *     b00 b01 b10 b11 t0 t1
     *
     * We did not conserve the matrix basis, so we must recompute
     * it now.
     */
    b00 = tmp;
    b01 = b00 + n;
    b10 = b01 + n;
    b11 = b10 + n;
    memmove(b11 + n, t0, n * 2 * sizeof * t0);
    t0 = b11 + n;
    t1 = t0 + n;
    smallints_to_fpr(b01, f, logn);
    smallints_to_fpr(b00, g, logn);
    smallints_to_fpr(b11, F, logn);
    smallints_to_fpr(b10, G, logn);
    PQCLEAN_FALCON512_NEON_FFT(b01, logn);
    PQCLEAN_FALCON512_NEON_FFT(b00, logn);
    PQCLEAN_FALCON512_NEON_FFT(b11, logn);
    PQCLEAN_FALCON512_NEON_FFT(b10, logn);
    PQCLEAN_FALCON512_NEON_poly_neg(b01, logn);
    PQCLEAN_FALCON512_NEON_poly_neg(b11, logn);
    tx = t1 + n;
    ty = tx + n;

    /*
     * Get the lattice point corresponding to that tiny vector.
     */
    memcpy(tx, t0, n * sizeof * t0);
    memcpy(ty, t1, n * sizeof * t1);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(tx, b00, logn);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, b10, logn);
    PQCLEAN_FALCON512_NEON_poly_add(tx, ty, logn);
    memcpy(ty, t0, n * sizeof * t0);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, b01, logn);

    memcpy(t0, tx, n * sizeof * tx);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(t1, b11, logn);
    PQCLEAN_FALCON512_NEON_poly_add(t1, ty, logn);
    PQCLEAN_FALCON512_NEON_iFFT(t0, logn);
    PQCLEAN_FALCON512_NEON_iFFT(t1, logn);

    s1tmp = (int16_t *)tx;
    sqn = 0;
    ng = 0;
    for (u = 0; u < n; u ++) {
        int32_t z;

        z = (int32_t)hm[u] - (int32_t)fpr_rint(t0[u]);
        sqn += (uint32_t)(z * z);
        ng |= sqn;
        s1tmp[u] = (int16_t)z;
    }
    sqn |= -(ng >> 31);

    /*
     * With "normal" degrees (e.g. 512 or 1024), it is very
     * improbable that the computed vector is not short enough;
     * however, it may happen in practice for the very reduced
     * versions (e.g. degree 16 or below). In that case, the caller
     * will loop, and we must not write anything into s2[] because
     * s2[] may overlap with the hashed message hm[] and we need
     * hm[] for the next iteration.
     */
    s2tmp = (int16_t *)tmp;
    for (u = 0; u < n; u ++) {
        s2tmp[u] = (int16_t) - fpr_rint(t1[u]);
    }
    if (PQCLEAN_FALCON512_NEON_is_short_half(sqn, s2tmp, logn)) {
        memcpy(s2, s2tmp, n * sizeof * s2);
        memcpy(tmp, s1tmp, n * sizeof * s1tmp);
        return 1;
    }
    return 0;
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_sign_dyn(int16_t *sig, inner_shake256_context *rng,
                                 const int8_t *f, const int8_t *g,
                                 const int8_t *F, const int8_t *G,
                                 const uint16_t *hm, unsigned logn, uint8_t *tmp) {
    fpr *ftmp;

    ftmp = (fpr *)tmp;
    for (;;) {
        /*
         * Signature produces short vectors s1 and s2. The
         * signature is acceptable only if the aggregate vector
         * s1,s2 is short; we must use the same bound as the
         * verifier.
         *
         * If the signature is acceptable, then we return only s2
         * (the verifier recomputes s1 from s2, the hashed message,
         * and the public key).
         */
        sampler_context spc;
        samplerZ samp;
        void *samp_ctx;

        /*
         * Normal sampling. We use a fast PRNG seeded from our
         * SHAKE context ('rng').
         */
        if (logn == 10) {
            spc.sigma_min = fpr_sigma_min_10;
        } else {
            spc.sigma_min = fpr_sigma_min_9;
        }
        PQCLEAN_FALCON512_NEON_prng_init(&spc.p, rng);
        samp = PQCLEAN_FALCON512_NEON_sampler;
        samp_ctx = &spc;

        /*
         * Do the actual signature.
         */
        if (do_sign_dyn(samp, samp_ctx, sig,
                        f, g, F, G, hm, logn, ftmp)) {
            break;
        }
    }
}
