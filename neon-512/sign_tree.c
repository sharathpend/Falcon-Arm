#include "inner.h"
#include "sign.h"
#include "macro.h"
#include "vfpr.h"
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
        tree0 = tree + 4;
        tree1 = tree + 8;

        float64x2x2_t tmp;
        float64x2_t a, b, c, c_re, c_im;
        float64x2_t w01, w02, w13, w23, x01;
        float64x2_t neon_i21, neon_1i2;
        int64x2_t scvt;
        double s_x0, s_x1, sigma;
        int64_t s_w0, s_w1, s_w2, s_w3;
        const double imagine[4] = {-1.0, 1.0, 1.0, -1.0};

        vload2(tmp, &t1[0]);
        a = tmp.val[0]; // a_re, a_im
        b = tmp.val[1]; // b_re, b_im
        vloadx2(tmp, &imagine[0]);
        neon_i21 = tmp.val[0];
        neon_1i2 = tmp.val[1];

        c = vfpr_add(a, b);
        w01 = vfpr_half(c);

        c = vfpr_sub(a, b);
        c_im = vmulq_f64(c, neon_i21);
        w23 = vpaddq_f64(c, c_im);
        w23 = vmulq_n_f64(w23, fpr_invsqrt8);

        x01 = w23;
        sigma = tree1[3];
        s_x0 = vgetq_lane_f64(x01, 0);
        s_x1 = vgetq_lane_f64(x01, 1);
#if DEBUG
        printv("w01", w01);
        printv("w23", w23);
#endif
#if FAKE_GAUSS
        s_w2 = gauss_test(s_x0, sigma);
        s_w3 = gauss_test(s_x1, sigma);
#else
        s_w2 = samp(samp_ctx, s_x0, sigma);
        s_w3 = samp(samp_ctx, s_x1, sigma);
#endif
        scvt = vsetq_lane_s64(s_w2, scvt, 0);
        scvt = vsetq_lane_s64(s_w3, scvt, 1);
        w23 = vcvtq_f64_s64(scvt);

        a = vsubq_f64(x01, w23);
        b = vld1q_f64(&tree1[0]);

        c_re = vmulq_f64(a, b);
        c_re = vmulq_f64(c_re, neon_1i2);

        b = vextq_f64(b, b, 1);
        c_im = vmulq_f64(a, b);
        c = vpaddq_f64(c_re, c_im);
        x01 = vaddq_f64(c, w01);

        sigma = tree1[2];
        s_x0 = vgetq_lane_f64(x01, 0);
        s_x1 = vgetq_lane_f64(x01, 1);
#if FAKE_GAUSS
        s_w0 = gauss_test(s_x0, sigma);
        s_w1 = gauss_test(s_x1, sigma);
#else
        s_w0 = samp(samp_ctx, s_x0, sigma);
        s_w1 = samp(samp_ctx, s_x1, sigma);
#endif
        scvt = vsetq_lane_s64(s_w0, scvt, 0);
        scvt = vsetq_lane_s64(s_w1, scvt, 1);
        w01 = vcvtq_f64_s64(scvt);

        a = w01;
        b = w23;
#if DEBUG
        printv("w01", w01);
        printv("w23", w23);
#endif
        c_re = vmulq_f64(b, neon_1i2);
        c = vpaddq_f64(c_re, b);
        c = vmulq_n_f64(c, fpr_invsqrt2);

        tmp.val[0] = vaddq_f64(a, c);
        tmp.val[1] = vsubq_f64(a, c);

        vst2q_f64(&z1[0], tmp);
#if DEBUG
        printv("z1-02", tmp.val[0]);
        printv("z1-13", tmp.val[1]);
#endif
        w02 = tmp.val[0];
        w13 = tmp.val[1];
        tmp = vld2q_f64(&t1[0]);

        w02 = vsubq_f64(tmp.val[0], w02);
        w13 = vsubq_f64(tmp.val[1], w13);

        tmp = vld2q_f64(&tree[0]);
        a = w02;
        b = tmp.val[0];

        c_re = vmulq_f64(a, b);
        c_re = vmulq_f64(c_re, neon_1i2);

        b = vextq_f64(b, b, 1);
        c_im = vmulq_f64(a, b);
        w02 = vpaddq_f64(c_re, c_im);

        a = w13;
        b = tmp.val[1];

        c_re = vmulq_f64(a, b);
        c_re = vmulq_f64(c_re, neon_1i2);

        b = vextq_f64(b, b, 1);
        c_im = vmulq_f64(a, b);
        w13 = vpaddq_f64(c_re, c_im);

#if DEBUG
        printv("w02", w02);
        printv("w13", w13);
#endif
        tmp = vld2q_f64(&t0[0]);
        w02 = vaddq_f64(w02, tmp.val[0]);
        w13 = vaddq_f64(w13, tmp.val[1]);

        /*
        * Second recursive invocation.
        */
        a = w02;
        b = w13;
        c = vaddq_f64(a, b);
        w01 = vfpr_half(w01);

        c = vsubq_f64(a, b);
        c_im = vmulq_f64(c, neon_i21);
        w23 = vpaddq_f64(c, c_im);
        w23 = vmulq_n_f64(w23, fpr_invsqrt8);

        x01 = w23;
        sigma = tree0[3];
        s_x0 = vgetq_lane_f64(x01, 0);
        s_x1 = vgetq_lane_f64(x01, 1);
#if FAKE_GAUSS
        s_w2 = gauss_test(s_x0, sigma);
        s_w3 = gauss_test(s_x1, sigma);
#else
        s_w2 = samp(samp_ctx, s_x0, sigma);
        s_w3 = samp(samp_ctx, s_x1, sigma);
#endif
        scvt = vsetq_lane_s64(s_w2, scvt, 0);
        scvt = vsetq_lane_s64(s_w3, scvt, 1);
        w23 = vcvtq_f64_s64(scvt);

        a = vsubq_f64(x01, w23);
        b = vld1q_f64(&tree0[0]);
        c_re = vmulq_f64(a, b);
        c_re = vmulq_f64(c_re, neon_1i2);
        b = vextq_f64(b, b, 1);
        c_im = vmulq_f64(a, b);
        c = vpaddq_f64(c_re, c_im);
        x01 = vaddq_f64(c, w01);
        sigma = tree0[2];

        s_x0 = vgetq_lane_f64(x01, 0);
        s_x1 = vgetq_lane_f64(x01, 1);
#if FAKE_GAUSS
        s_w0 = gauss_test(s_x0, sigma);
        s_w1 = gauss_test(s_x1, sigma);
#else
        s_w0 = samp(samp_ctx, s_x0, sigma);
        s_w1 = samp(samp_ctx, s_x1, sigma);
#endif

        scvt = vsetq_lane_s64(s_w0, scvt, 0);
        scvt = vsetq_lane_s64(s_w1, scvt, 1);
        a = vcvtq_f64_s64(scvt);

        c_im = w23;
        c_re = vmulq_f64(w23, neon_1i2);
        c = vpaddq_f64(c_re, c_im);
        c = vmulq_n_f64(c, fpr_invsqrt2);

        tmp.val[0] = vaddq_f64(a, c);
        tmp.val[1] = vsubq_f64(a, c);

        vst2q_f64(&z0[0], tmp);

        return;
    }

    /*
     * Case logn == 1 is reachable only when using Falcon-2 (the
     * smallest size for which Falcon is mathematically defined, but
     * of course way too insecure to be of any use).
     */
    if (logn == 1) {
        fpr sigma;
        float64x2_t x01, y01, a, b, c, c_re, c_im, neon_1i2;
        fpr s_x0, s_x1;
        int64x2_t scvt;
        int64_t y0, y1;
        const double imagine[2] = {1.0, -1.0};

        neon_1i2 = vld1q_f64(&imagine[0]);
        x01 = vld1q_f64(&t1[0]);
        sigma = tree[3];
        s_x0 = vgetq_lane_f64(x01, 0);
        s_x1 = vgetq_lane_f64(x01, 1);
#if FAKE_GAUSS
        y0 = gauss_test(s_x0, sigma);
        y1 = gauss_test(s_x1, sigma);
#else
        y0 = samp(samp_ctx, s_x0, sigma);
        y1 = samp(samp_ctx, s_x1, sigma);
#endif
        scvt = vsetq_lane_s64(y0, scvt, 0);
        scvt = vsetq_lane_s64(y1, scvt, 1);
        y01 = vcvtq_f64_s64(scvt);
        vst1q_f64(&z1[0], y01);

        a = vsubq_f64(x01, y01);
        b = vld1q_f64(&tree[0]);
        c_re = vmulq_f64(a, b);
        c_re = vmulq_f64(c_re, neon_1i2);

        b = vextq_f64(b, b, 1);
        c_im = vmulq_f64(a, b);
        c = vpaddq_f64(c_re, c_im);
        x01 = vld1q_f64(&t0[0]);
        x01 = vaddq_f64(c, x01);
        sigma = tree[2];

        s_x0 = vgetq_lane_f64(x01, 0);
        s_x1 = vgetq_lane_f64(x01, 1);
#if FAKE_GAUSS
        y0 = gauss_test(s_x0, sigma);
        y1 = gauss_test(s_x1, sigma);
#else
        y0 = samp(samp_ctx, s_x0, sigma);
        y1 = samp(samp_ctx, s_x1, sigma);
#endif
        scvt = vsetq_lane_s64(y0, scvt, 0);
        scvt = vsetq_lane_s64(y1, scvt, 1);
        y01 = vcvtq_f64_s64(scvt);
        vst1q_f64(&z0[0], y01);

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

    fpr *t0, *t1, *tx, *ty;
    const fpr *b00, *b01, *b10, *b11, *tree;
    fpr ni;
    uint32_t sqn, ng;
    int16_t *s1tmp, *s2tmp;

    t0 = tmp;
    t1 = t0 + FALCON_N;
    b00 = expanded_key + skoff_b00(logn);
    b01 = expanded_key + skoff_b01(logn);
    b10 = expanded_key + skoff_b10(logn);
    b11 = expanded_key + skoff_b11(logn);
    tree = expanded_key + skoff_tree(logn);

    uint64x2x4_t tmp_u64;
    float64x2x4_t tmp_fpr;

    /*
     * Set the target vector to [hm, 0] (hm is the hashed message).
     */
    for (size_t u = 0; u < FALCON_N; u +=8) {
        tmp_u64 = vld1q_u64_x4(&hm[u]);
        tmp_fpr.val[0] = vfpr_of(tmp_u64.val[0]);
        tmp_fpr.val[1] = vfpr_of(tmp_u64.val[1]);
        tmp_fpr.val[2] = vfpr_of(tmp_u64.val[2]);
        tmp_fpr.val[3] = vfpr_of(tmp_u64.val[3]);
        vst1q_f64_x4(&t0[u], tmp_fpr);
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
    memcpy(t1, t0, FALCON_N * sizeof * t0);
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

    tx = t1 + FALCON_N;
    ty = tx + FALCON_N;

    /*
     * Apply sampling. Output is written back in [tx, ty].
     */
    ffSampling_fft(samp, samp_ctx, tx, ty, tree, t0, t1, logn, ty + FALCON_N);

    /*
     * Get the lattice point corresponding to that tiny vector.
     */
    memcpy(t0, tx, FALCON_N * sizeof * tx);
    memcpy(t1, ty, FALCON_N * sizeof * ty);
    // tx = tx * b00
    // ty = ty * b10 
    // tx = tx + ty 
    
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, b10, logn);
    // PQCLEAN_FALCON512_NEON_poly_add(tx, ty, logn);
    // tx = tx * b00 
    // tx = tx + ty * b10 = tx * b00 + ty * b10
    PQCLEAN_FALCON512_NEON_poly_mul_fft(tx, tx, b00);
    PQCLEAN_FALCON512_NEON_poly_mul_fft_add(tx, tx, ty, b10);
    memcpy(ty, t0, FALCON_N * sizeof * t0);
    
    // ty = ty * b01 
    // t1 = t1 * b11
    // t1 = t1 + ty 
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, b01, logn);
    // PQCLEAN_FALCON512_NEON_poly_mul_fft(t1, b11, logn);
    // PQCLEAN_FALCON512_NEON_poly_add(t1, ty, logn);
    PQCLEAN_FALCON512_NEON_poly_mul_fft(ty, ty, b01);
    PQCLEAN_FALCON512_NEON_poly_mul_fft_add(t1, ty, t1, b11);

    memcpy(t0, tx, FALCON_N * sizeof * tx);
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
    for (size_t u = 0; u < FALCON_N; u ++) {
        int32_t z;

        z = (int32_t)hm[u] - (int32_t)fpr_rint(t0[u]);
        sqn += (uint32_t)(z * z);
        ng |= sqn;
        s1tmp[u] = (int16_t)z;
    }
    sqn |= -(ng >> 31);

    /* // Total SIMD register: 16
    float64x2x4_t t0tmp[4]; // 16
    int64x2x4_t tmps64[4];
    int32x4x2_t tmps32[4];
    int16x8x4_t tmps16;
    int32x4x4_t z[2];
    uint32x4x4_t sqn[2];
    uint32x4_t sqn_sum[2];
    uint32x4_t ng;

    for (size_t u = 0; u < FALCON_N; u +=32) {
        t0tmp[0] = vld1q_f64_x4(&t0[u]);
        t0tmp[1] = vld1q_f64_x4(&t0[u+8]);
        t0tmp[2] = vld1q_f64_x4(&t0[u+16]);
        t0tmp[3] = vld1q_f64_x4(&t0[u+24]);
        tmps16 = vld1q_s16_x4(&hm[u]);


        tmps64[0].val[0] = vfpr_rint(t0tmp[0].val[0]);
        tmps64[0].val[1] = vfpr_rint(t0tmp[0].val[1]);
        tmps64[0].val[2] = vfpr_rint(t0tmp[0].val[2]);
        tmps64[0].val[3] = vfpr_rint(t0tmp[0].val[3]);

        tmps64[1].val[0] = vfpr_rint(t0tmp[1].val[0]);
        tmps64[1].val[1] = vfpr_rint(t0tmp[1].val[1]);
        tmps64[1].val[2] = vfpr_rint(t0tmp[1].val[2]);
        tmps64[1].val[3] = vfpr_rint(t0tmp[1].val[3]);

        tmps64[2].val[0] = vfpr_rint(t0tmp[2].val[0]);
        tmps64[2].val[1] = vfpr_rint(t0tmp[2].val[1]);
        tmps64[2].val[2] = vfpr_rint(t0tmp[2].val[2]);
        tmps64[2].val[3] = vfpr_rint(t0tmp[2].val[3]);

        tmps64[3].val[0] = vfpr_rint(t0tmp[3].val[0]);
        tmps64[3].val[1] = vfpr_rint(t0tmp[3].val[1]);
        tmps64[3].val[2] = vfpr_rint(t0tmp[3].val[2]);
        tmps64[3].val[3] = vfpr_rint(t0tmp[3].val[3]);

        tmps32[0].val[0] = vmovn_s64(tmps64[0].val[0]);
        tmps32[0].val[1] = vmovn_s64(tmps64[0].val[2]);
        tmps32[0].val[0] = vmovn_high_s64(tmps32[0].val[0], tmps64[0].val[1]);
        tmps32[0].val[1] = vmovn_high_s64(tmps32[0].val[1], tmps64[0].val[3]);

        tmps32[1].val[0] = vmovn_s64(tmps64[1].val[0]);
        tmps32[1].val[1] = vmovn_s64(tmps64[1].val[2]);
        tmps32[1].val[0] = vmovn_high_s64(tmps32[1].val[0], tmps64[1].val[1]);
        tmps32[1].val[1] = vmovn_high_s64(tmps32[1].val[1], tmps64[1].val[3]);

        tmps32[2].val[0] = vmovn_s64(tmps64[2].val[0]);
        tmps32[2].val[1] = vmovn_s64(tmps64[2].val[2]);
        tmps32[2].val[0] = vmovn_high_s64(tmps32[2].val[0], tmps64[2].val[1]);
        tmps32[2].val[1] = vmovn_high_s64(tmps32[2].val[1], tmps64[2].val[3]);

        tmps32[3].val[0] = vmovn_s64(tmps64[3].val[0]);
        tmps32[3].val[1] = vmovn_s64(tmps64[3].val[2]);
        tmps32[3].val[0] = vmovn_high_s64(tmps32[3].val[0], tmps64[3].val[1]);
        tmps32[3].val[1] = vmovn_high_s64(tmps32[3].val[1], tmps64[3].val[3]);

        z[0].val[0] = vsubw_s16(tmps32[0].val[0], vget_low_s16(tmps16.val[0]));
        z[0].val[1] = vsubw_s16(tmps32[1].val[0], vget_low_s16(tmps16.val[1]));
        z[0].val[2] = vsubw_s16(tmps32[2].val[0], vget_low_s16(tmps16.val[2]));
        z[0].val[3] = vsubw_s16(tmps32[3].val[0], vget_low_s16(tmps16.val[3]));

        z[1].val[0] = vsubw_high_s16(tmps32[0].val[1], tmps16.val[0]);
        z[1].val[1] = vsubw_high_s16(tmps32[1].val[1], tmps16.val[1]);
        z[1].val[2] = vsubw_high_s16(tmps32[2].val[1], tmps16.val[2]);
        z[1].val[3] = vsubw_high_s16(tmps32[3].val[1], tmps16.val[3]);

        sqn[0].val[0] = vmulq_s32(z[0].val[0], z[0].val[0]);
        sqn[0].val[1] = vmulq_s32(z[0].val[1], z[0].val[1]);
        sqn[0].val[2] = vmulq_s32(z[0].val[2], z[0].val[2]);
        sqn[0].val[3] = vmulq_s32(z[0].val[3], z[0].val[3]);

        sqn[1].val[0] = vmulq_s32(z[1].val[0], z[1].val[0]);
        sqn[1].val[1] = vmulq_s32(z[1].val[1], z[1].val[1]);
        sqn[1].val[2] = vmulq_s32(z[1].val[2], z[1].val[2]);
        sqn[1].val[3] = vmulq_s32(z[1].val[3], z[1].val[3]);

        sqn_sum[0] = vaddq_s32(sqn_sum[0], sqn[0].val[0]);
        sqn_sum[0] = vaddq_s32(sqn_sum[0], sqn[0].val[1]);
        sqn_sum[0] = vaddq_s32(sqn_sum[0], sqn[0].val[2]);
        sqn_sum[0] = vaddq_s32(sqn_sum[0], sqn[0].val[3]);

        sqn_sum[1] = vaddq_s32(sqn_sum[1], sqn[1].val[0]);
        sqn_sum[1] = vaddq_s32(sqn_sum[1], sqn[1].val[1]);
        sqn_sum[1] = vaddq_s32(sqn_sum[1], sqn[1].val[2]);
        sqn_sum[1] = vaddq_s32(sqn_sum[1], sqn[1].val[3]);

        ng = vpaddq_s32(sqn_sum[0], sqn_sum[1]);
    }
 */
    ///////////

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
    for (size_t u = 0; u < FALCON_N; u ++) {
        s2tmp[u] = (int16_t) - fpr_rint(t1[u]);
    }
    /* 
    float64x2x4_t t1tmp[4]; // 16
    int64x2x4_t s2tmps64[4];
    int32x4x2_t s2tmps32[4];
    int16x8x4_t s2tmps16;
    for (size_t u = 0; u < FALCON_N; u+= 32)
    {
        t1tmp[0] = vld1q_f64_x4(&t1[u + 0]);
        t1tmp[1] = vld1q_f64_x4(&t1[u + 8]);
        t1tmp[2] = vld1q_f64_x4(&t1[u + 16]);
        t1tmp[3] = vld1q_f64_x4(&t1[u + 24]);

        s2tmps64[0].val[0] = vfpr_rint(t1tmp[0].val[0]);
        s2tmps64[0].val[1] = vfpr_rint(t1tmp[0].val[1]);
        s2tmps64[0].val[2] = vfpr_rint(t1tmp[0].val[2]);
        s2tmps64[0].val[3] = vfpr_rint(t1tmp[0].val[3]);

        s2tmps64[1].val[0] = vfpr_rint(t1tmp[1].val[0]);
        s2tmps64[1].val[1] = vfpr_rint(t1tmp[1].val[1]);
        s2tmps64[1].val[2] = vfpr_rint(t1tmp[1].val[2]);
        s2tmps64[1].val[3] = vfpr_rint(t1tmp[1].val[3]);

        s2tmps64[2].val[0] = vfpr_rint(t1tmp[2].val[0]);
        s2tmps64[2].val[1] = vfpr_rint(t1tmp[2].val[1]);
        s2tmps64[2].val[2] = vfpr_rint(t1tmp[2].val[2]);
        s2tmps64[2].val[3] = vfpr_rint(t1tmp[2].val[3]);

        s2tmps64[3].val[0] = vfpr_rint(t1tmp[3].val[0]);
        s2tmps64[3].val[1] = vfpr_rint(t1tmp[3].val[1]);
        s2tmps64[3].val[2] = vfpr_rint(t1tmp[3].val[2]);
        s2tmps64[3].val[3] = vfpr_rint(t1tmp[3].val[3]);

        s2tmps32[0].val[0] = vmovn_s64(s2tmps64[0].val[0]);
        s2tmps32[0].val[1] = vmovn_s64(s2tmps64[0].val[2]);
        s2tmps32[0].val[0] = vmovn_high_s64(s2tmps32[0].val[0], s2tmps64[0].val[1]);
        s2tmps32[0].val[1] = vmovn_high_s64(s2tmps32[0].val[1], s2tmps64[0].val[3]);

        s2tmps32[1].val[0] = vmovn_s64(s2tmps64[1].val[0]);
        s2tmps32[1].val[1] = vmovn_s64(s2tmps64[1].val[2]);
        s2tmps32[1].val[0] = vmovn_high_s64(s2tmps32[1].val[0], s2tmps64[1].val[1]);
        s2tmps32[1].val[1] = vmovn_high_s64(s2tmps32[1].val[1], s2tmps64[1].val[3]);

        s2tmps32[2].val[0] = vmovn_s64(s2tmps64[2].val[0]);
        s2tmps32[2].val[1] = vmovn_s64(s2tmps64[2].val[2]);
        s2tmps32[2].val[0] = vmovn_high_s64(s2tmps32[2].val[0], s2tmps64[2].val[1]);
        s2tmps32[2].val[1] = vmovn_high_s64(s2tmps32[2].val[1], s2tmps64[2].val[3]);

        s2tmps32[3].val[0] = vmovn_s64(s2tmps64[3].val[0]);
        s2tmps32[3].val[1] = vmovn_s64(s2tmps64[3].val[2]);
        s2tmps32[3].val[0] = vmovn_high_s64(s2tmps32[3].val[0], s2tmps64[3].val[1]);
        s2tmps32[3].val[1] = vmovn_high_s64(s2tmps32[3].val[1], s2tmps64[3].val[3]);

        s2tmps16.val[0] = vmovn_s32(s2tmps32[0].val[0]);
        s2tmps16.val[1] = vmovn_s32(s2tmps32[1].val[0]);
        s2tmps16.val[2] = vmovn_s32(s2tmps32[2].val[0]);
        s2tmps16.val[3] = vmovn_s32(s2tmps32[3].val[0]);
        s2tmps16.val[0] = vmovn_high_s32(s2tmps16.val[0], s2tmps32[0].val[1]);
        s2tmps16.val[1] = vmovn_high_s32(s2tmps16.val[1], s2tmps32[1].val[1]);
        s2tmps16.val[2] = vmovn_high_s32(s2tmps16.val[2], s2tmps32[2].val[1]);
        s2tmps16.val[3] = vmovn_high_s32(s2tmps16.val[3], s2tmps32[3].val[1]);
        
        vst1q_s16_x4(&s2tmp[u], s2tmps16);
    } */

    if (PQCLEAN_FALCON512_NEON_is_short_half(sqn, s2tmp, logn)) {
        memcpy(s2, s2tmp, FALCON_N * sizeof * s2);
        memcpy(tmp, s1tmp, FALCON_N * sizeof * s1tmp);
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
