#include "inner.h"
#include "sign.h"
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
 * Perform Fast Fourier Sampling for target vector t and LDL tree T.
 * tmp[] must have size for at least two polynomials of size 2^logn.
 */
static void
ffSampling_fft(samplerZ samp, void *samp_ctx,
               fpr *z0, fpr *z1,
               const fpr *tree,
               const fpr *t0, const fpr *t1, unsigned logn,
               fpr *tmp) {
    size_t n, hn;
    const fpr *tree0, *tree1;

    /*
     * When logn == 2, we inline the last two recursion levels.
     */
    if (logn == 2) {
        fpr x0, x1, y0, y1, w0, w1, w2, w3, sigma;
        fpr a_re, a_im, b_re, b_im, c_re, c_im;

        tree0 = tree + 4;
        tree1 = tree + 8;

        /*
         * We split t1 into w*, then do the recursive invocation,
         * with output in w*. We finally merge back into z1.
         */
        a_re = t1[0];
        a_im = t1[2];
        b_re = t1[1];
        b_im = t1[3];
        c_re = fpr_add(a_re, b_re);
        c_im = fpr_add(a_im, b_im);
        w0 = fpr_half(c_re);
        w1 = fpr_half(c_im);
        c_re = fpr_sub(a_re, b_re);
        c_im = fpr_sub(a_im, b_im);
        w2 = fpr_mul(fpr_add(c_re, c_im), fpr_invsqrt8);
        w3 = fpr_mul(fpr_sub(c_im, c_re), fpr_invsqrt8);

        x0 = w2;
        x1 = w3;
        sigma = tree1[3];
        w2 = fpr_of(samp(samp_ctx, x0, sigma));
        w3 = fpr_of(samp(samp_ctx, x1, sigma));
        a_re = fpr_sub(x0, w2);
        a_im = fpr_sub(x1, w3);
        b_re = tree1[0];
        b_im = tree1[1];
        c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
        c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
        x0 = fpr_add(c_re, w0);
        x1 = fpr_add(c_im, w1);
        sigma = tree1[2];
        w0 = fpr_of(samp(samp_ctx, x0, sigma));
        w1 = fpr_of(samp(samp_ctx, x1, sigma));

        a_re = w0;
        a_im = w1;
        b_re = w2;
        b_im = w3;
        c_re = fpr_mul(fpr_sub(b_re, b_im), fpr_invsqrt2);
        c_im = fpr_mul(fpr_add(b_re, b_im), fpr_invsqrt2);
        z1[0] = w0 = fpr_add(a_re, c_re);
        z1[2] = w2 = fpr_add(a_im, c_im);
        z1[1] = w1 = fpr_sub(a_re, c_re);
        z1[3] = w3 = fpr_sub(a_im, c_im);

        /*
         * Compute tb0 = t0 + (t1 - z1) * L. Value tb0 ends up in w*.
         */
        w0 = fpr_sub(t1[0], w0);
        w1 = fpr_sub(t1[1], w1);
        w2 = fpr_sub(t1[2], w2);
        w3 = fpr_sub(t1[3], w3);

        a_re = w0;
        a_im = w2;
        b_re = tree[0];
        b_im = tree[2];
        w0 = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
        w2 = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
        a_re = w1;
        a_im = w3;
        b_re = tree[1];
        b_im = tree[3];
        w1 = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
        w3 = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));

        w0 = fpr_add(w0, t0[0]);
        w1 = fpr_add(w1, t0[1]);
        w2 = fpr_add(w2, t0[2]);
        w3 = fpr_add(w3, t0[3]);

        /*
         * Second recursive invocation.
         */
        a_re = w0;
        a_im = w2;
        b_re = w1;
        b_im = w3;
        c_re = fpr_add(a_re, b_re);
        c_im = fpr_add(a_im, b_im);
        w0 = fpr_half(c_re);
        w1 = fpr_half(c_im);
        c_re = fpr_sub(a_re, b_re);
        c_im = fpr_sub(a_im, b_im);
        w2 = fpr_mul(fpr_add(c_re, c_im), fpr_invsqrt8);
        w3 = fpr_mul(fpr_sub(c_im, c_re), fpr_invsqrt8);

        x0 = w2;
        x1 = w3;
        sigma = tree0[3];
        w2 = y0 = fpr_of(samp(samp_ctx, x0, sigma));
        w3 = y1 = fpr_of(samp(samp_ctx, x1, sigma));
        a_re = fpr_sub(x0, y0);
        a_im = fpr_sub(x1, y1);
        b_re = tree0[0];
        b_im = tree0[1];
        c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
        c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
        x0 = fpr_add(c_re, w0);
        x1 = fpr_add(c_im, w1);
        sigma = tree0[2];
        w0 = fpr_of(samp(samp_ctx, x0, sigma));
        w1 = fpr_of(samp(samp_ctx, x1, sigma));

        a_re = w0;
        a_im = w1;
        b_re = w2;
        b_im = w3;
        c_re = fpr_mul(fpr_sub(b_re, b_im), fpr_invsqrt2);
        c_im = fpr_mul(fpr_add(b_re, b_im), fpr_invsqrt2);
        z0[0] = fpr_add(a_re, c_re);
        z0[2] = fpr_add(a_im, c_im);
        z0[1] = fpr_sub(a_re, c_re);
        z0[3] = fpr_sub(a_im, c_im);

        return;
    }

    /*
     * Case logn == 1 is reachable only when using Falcon-2 (the
     * smallest size for which Falcon is mathematically defined, but
     * of course way too insecure to be of any use).
     */
    if (logn == 1) {
        fpr x0, x1, y0, y1, sigma;
        fpr a_re, a_im, b_re, b_im, c_re, c_im;

        x0 = t1[0];
        x1 = t1[1];
        sigma = tree[3];
        z1[0] = y0 = fpr_of(samp(samp_ctx, x0, sigma));
        z1[1] = y1 = fpr_of(samp(samp_ctx, x1, sigma));
        a_re = fpr_sub(x0, y0);
        a_im = fpr_sub(x1, y1);
        b_re = tree[0];
        b_im = tree[1];
        c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
        c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
        x0 = fpr_add(c_re, t0[0]);
        x1 = fpr_add(c_im, t0[1]);
        sigma = tree[2];
        z0[0] = fpr_of(samp(samp_ctx, x0, sigma));
        z0[1] = fpr_of(samp(samp_ctx, x1, sigma));

        return;
    }

    /*
     * Normal end of recursion is for logn == 0. Since the last
     * steps of the recursions were inlined in the blocks above
     * (when logn == 1 or 2), this case is not reachable, and is
     * retained here only for documentation purposes.

    if (logn == 0) {
        fpr x0, x1, sigma;

        x0 = t0[0];
        x1 = t1[0];
        sigma = tree[0];
        z0[0] = fpr_of(samp(samp_ctx, x0, sigma));
        z1[0] = fpr_of(samp(samp_ctx, x1, sigma));
        return;
    }

     */

    /*
     * General recursive case (logn >= 3).
     */

    n = (size_t)1 << logn;
    hn = n >> 1;
    tree0 = tree + n;
    tree1 = tree + n + ffLDL_treesize(logn - 1);


    /* 
    n = 512 : 9 levels 
    r: 256 l: 256 

    level 3, 4, [5 - 8]

    level 1 
    r: 2   l: 2 <------ 
     */
    /*
     * We split t1 into z1 (reused as temporary storage), then do
     * the recursive invocation, with output in tmp. We finally
     * merge back into z1.
     */
    PQCLEAN_FALCON512_NEON_poly_split_fft(z1, z1 + hn, t1, logn);
    ffSampling_fft(samp, samp_ctx, tmp, tmp + hn,
                   tree1, z1, z1 + hn, logn - 1, tmp + n);
    PQCLEAN_FALCON512_NEON_poly_merge_fft(z1, tmp, tmp + hn, logn);

    /*
     * Compute tb0 = t0 + (t1 - z1) * L. Value tb0 ends up in tmp[].
     */
    memcpy(tmp, t1, n * sizeof * t1);
    PQCLEAN_FALCON512_NEON_poly_sub(tmp, z1, logn);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(tmp, tree, logn);
    PQCLEAN_FALCON512_NEON_poly_add(tmp, t0, logn);

    /*
     * Second recursive invocation.
     */
    PQCLEAN_FALCON512_NEON_poly_split_fft(z0, z0 + hn, tmp, logn);
    ffSampling_fft(samp, samp_ctx, tmp, tmp + hn,
                   tree0, z0, z0 + hn, logn - 1, tmp + n);
    PQCLEAN_FALCON512_NEON_poly_merge_fft(z0, tmp, tmp + hn, logn);
}

/*
 * Compute a signature: the signature contains two vectors, s1 and s2.
 * The s1 vector is not returned. The squared norm of (s1,s2) is
 * computed, and if it is short enough, then s2 is returned into the
 * s2[] buffer, and 1 is returned; otherwise, s2[] is untouched and 0 is
 * returned; the caller should then try again. This function uses an
 * expanded key.
 *
 * tmp[] must have room for at least six polynomials.
 */
static int
do_sign_tree(samplerZ samp, void *samp_ctx, int16_t *s2,
             const fpr *expanded_key,
             const uint16_t *hm,
             unsigned logn, fpr *tmp) {
    size_t n, u;
    fpr *t0, *t1, *tx, *ty;
    const fpr *b00, *b01, *b10, *b11, *tree;
    fpr ni;
    uint32_t sqn, ng;
    int16_t *s1tmp, *s2tmp;

    n = MKN(logn);
    t0 = tmp;
    t1 = t0 + n;
    b00 = expanded_key + skoff_b00(logn);
    b01 = expanded_key + skoff_b01(logn);
    b10 = expanded_key + skoff_b10(logn);
    b11 = expanded_key + skoff_b11(logn);
    tree = expanded_key + skoff_tree(logn);

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
    // t0 = FFT(t0)
    // PQCLEAN_FALCON512_NEON_FFT(t0, logn);
    PQCLEAN_FALCON512_NEON_FFT(t0, false);
    ni = fpr_inverse_of_q;
    // t1 = t0
    memcpy(t1, t0, n * sizeof * t0);
    // t1 = t1 * b01 
    // t1 = t1 * (-ni)
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(t1, b01, logn);
    // PQCLEAN_FALCON512_NEON_poly_mulconst(t1, fpr_neg(ni), logn);
    PQCLEAN_FALCON512_NEON_poly_mul_fftconst(t1, t1, b01, fpr_neg(ni));
    // t0 = t0 * b11
    // t0 = t0 * ni
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(t0, b11, logn);
    // PQCLEAN_FALCON512_NEON_poly_mulconst(t0, ni, logn);
    PQCLEAN_FALCON512_NEON_poly_mul_fftconst(t0, t0, b11, ni);

    tx = t1 + n;
    ty = tx + n;

    /*
     * Apply sampling. Output is written back in [tx, ty].
     */
    ffSampling_fft(samp, samp_ctx, tx, ty, tree, t0, t1, logn, ty + n);

    /*
     * Get the lattice point corresponding to that tiny vector.
     */
    memcpy(t0, tx, n * sizeof * tx);
    memcpy(t1, ty, n * sizeof * ty);
    // tx = tx * b00
    // ty = ty * b10 
    // tx = tx + ty 
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(tx, b00, logn);
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, b10, logn);
    // PQCLEAN_FALCON512_NEON_poly_add(tx, ty, logn);
    // tx = tx * b00 
    // tx = tx + ty * b10 = tx * b00 + ty * b10
    PQCLEAN_FALCON512_NEON_poly_mul_fft(tx, tx, b00);
    PQCLEAN_FALCON512_NEON_poly_mul_fft_add(tx, tx, ty, b10);
    memcpy(ty, t0, n * sizeof * t0);
    
    // ty = ty * b01 
    // t1 = t1 * b11
    // t1 = t1 + ty 
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, b01, logn);
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(t1, b11, logn);
    // PQCLEAN_FALCON512_NEON_poly_add(t1, ty, logn);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, ty, b01);
    PQCLEAN_FALCON512_NEON_poly_mul_fft_add(t1, ty, t1, b11);

    memcpy(t0, tx, n * sizeof * tx);
    // t0 = iFFT(t0)
    // t1 = iFFT(t1)
    PQCLEAN_FALCON512_NEON_iFFT(t0);
    PQCLEAN_FALCON512_NEON_iFFT(t1);

    /*
     * Compute the signature.
     */
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
PQCLEAN_FALCON512_NEON_sign_tree(int16_t *sig, inner_shake256_context *rng,
                                  const fpr *expanded_key,
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
        if (do_sign_tree(samp, samp_ctx, sig,
                         expanded_key, hm, logn, ftmp)) {
            break;
        }
    }
}
