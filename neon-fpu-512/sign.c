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

#include "inner.h"
#include "macrofx4.h"
#include "macrof.h"
#include <arm_neon.h>
#include "util.h"
#include <stdio.h>
/* =================================================================== */

/*
 * Compute degree N from logarithm 'logn'.
 */
#define MKN(logn)   ((size_t)1 << (logn))

/* =================================================================== */
/*
 * Binary case:
 *   N = 2^logn
 *   phi = X^N+1
 */

/*
 * Get the size of the LDL tree for an input with polynomials of size
 * 2^logn. The size is expressed in the number of elements.
 */
static inline unsigned
ffLDL_treesize(unsigned logn)
{
	/*
	 * For logn = 0 (polynomials are constant), the "tree" is a
	 * single element. Otherwise, the tree node has size 2^logn, and
	 * has two child trees for size logn-1 each. Thus, treesize s()
	 * must fulfill these two relations:
	 *
	 *   s(0) = 1
	 *   s(logn) = (2^logn) + 2*s(logn-1)
	 */
	return (logn + 1) << logn;
}

/*
 * Inner function for ffLDL_fft(). It expects the matrix to be both
 * auto-adjoint and quasicyclic; also, it uses the source operands
 * as modifiable temporaries.
 *
 * tmp[] must have room for at least one polynomial.
 */
static void
ffLDL_fft_inner(fpr *restrict tree,
	fpr *restrict g0, fpr *restrict g1, unsigned logn, fpr *restrict tmp)
{
	size_t n, hn;

	n = MKN(logn);
	if (n == 1) {
		tree[0] = g0[0];
		return;
	}
	hn = n >> 1;

	/*
	 * The LDL decomposition yields L (which is written in the tree)
	 * and the diagonal of D. Since d00 = g0, we just write d11
	 * into tmp.
	 */
	Zf(poly_LDLmv_fft)(tmp, tree, g0, g1, g0, logn);

	/*
	 * Split d00 (currently in g0) and d11 (currently in tmp). We
	 * reuse g0 and g1 as temporary storage spaces:
	 *   d00 splits into g1, g1+hn
	 *   d11 splits into g0, g0+hn
	 */
	Zf(poly_split_fft)(g1, g1 + hn, g0, logn);
	Zf(poly_split_fft)(g0, g0 + hn, tmp, logn);

	/*
	 * Each split result is the first row of a new auto-adjoint
	 * quasicyclic matrix for the next recursive step.
	 */
	ffLDL_fft_inner(tree + n,
		g1, g1 + hn, logn - 1, tmp);
	ffLDL_fft_inner(tree + n + ffLDL_treesize(logn - 1),
		g0, g0 + hn, logn - 1, tmp);
}

/*
 * Compute the ffLDL tree of an auto-adjoint matrix G. The matrix
 * is provided as three polynomials (FFT representation).
 *
 * The "tree" array is filled with the computed tree, of size
 * (logn+1)*(2^logn) elements (see ffLDL_treesize()).
 *
 * Input arrays MUST NOT overlap, except possibly the three unmodified
 * arrays g00, g01 and g11. tmp[] should have room for at least three
 * polynomials of 2^logn elements each.
 */
static void
ffLDL_fft(fpr *restrict tree, const fpr *restrict g00,
	const fpr *restrict g01, const fpr *restrict g11,
	unsigned logn, fpr *restrict tmp)
{
	size_t n, hn;
	fpr *d00, *d11;

	n = MKN(logn);
	if (n == 1) {
		tree[0] = g00[0];
		return;
	}
	hn = n >> 1;
	d00 = tmp;
	d11 = tmp + n;
	tmp += n << 1;

	memcpy(d00, g00, n * sizeof *g00);
	Zf(poly_LDLmv_fft)(d11, tree, g00, g01, g11, logn);

	Zf(poly_split_fft)(tmp, tmp + hn, d00, logn);
	Zf(poly_split_fft)(d00, d00 + hn, d11, logn);
	memcpy(d11, tmp, n * sizeof *tmp);
	ffLDL_fft_inner(tree + n,
		d11, d11 + hn, logn - 1, tmp);
	ffLDL_fft_inner(tree + n + ffLDL_treesize(logn - 1),
		d00, d00 + hn, logn - 1, tmp);
}

/*
 * Normalize an ffLDL tree: each leaf of value x is replaced with
 * sigma / sqrt(x).
 */
static void
ffLDL_binary_normalize(fpr *tree, unsigned orig_logn, unsigned logn)
{
	/*
	 * TODO: make an iterative version.
	 */
	size_t n;

	n = MKN(logn);
	if (n == 1) {
		/*
		 * We actually store in the tree leaf the inverse of
		 * the value mandated by the specification: this
		 * saves a division both here and in the sampler.
		 */
        tree[0] = fpr_mul(fpr_sqrt(tree[0]), fpr_inv_sigma);
	} else {
		ffLDL_binary_normalize(tree + n, orig_logn, logn - 1);
		ffLDL_binary_normalize(tree + n + ffLDL_treesize(logn - 1),
			orig_logn, logn - 1);
	}
}

/* =================================================================== */

/*
 * The expanded private key contains:
 *  - The B0 matrix (four elements)
 *  - The ffLDL tree
 */

static inline size_t
skoff_b00(unsigned logn)
{
	(void)logn;
	return 0;
}

static inline size_t
skoff_b01(unsigned logn)
{
	return MKN(logn);
}

static inline size_t
skoff_b10(unsigned logn)
{
	return 2 * MKN(logn);
}

static inline size_t
skoff_b11(unsigned logn)
{
	return 3 * MKN(logn);
}

static inline size_t
skoff_tree(unsigned logn)
{
	return 4 * MKN(logn);
}

/* see inner.h */
void
Zf(expand_privkey)(fpr *restrict expanded_key,
	const int8_t *f, const int8_t *g,
	const int8_t *F, const int8_t *G,
	unsigned logn, uint8_t *restrict tmp)
{
	fpr *rf, *rg, *rF, *rG;
	fpr *b00, *b01, *b10, *b11;
	fpr *g00, *g01, *g11, *gxx;
	fpr *tree;
	const unsigned n = MKN(logn);

	b00 = expanded_key + skoff_b00(logn);
	b01 = expanded_key + skoff_b01(logn);
	b10 = expanded_key + skoff_b10(logn);
	b11 = expanded_key + skoff_b11(logn);
	tree = expanded_key + skoff_tree(logn);

	/*
	 * We load the private key elements directly into the B0 matrix,
	 * since B0 = [[g, -f], [G, -F]].
	 */
	rf = b01;
	rg = b00;
	rF = b11;
	rG = b10;

	smallints_to_fpr(rf, f, logn);
	smallints_to_fpr(rg, g, logn);
	smallints_to_fpr(rF, F, logn);
	smallints_to_fpr(rG, G, logn);

	/*
	 * Compute the FFT for the key elements, and negate f and F.
	 */
	Zf(FFT)(rf, logn);
	Zf(FFT)(rF, logn);
    Zf(poly_neg)(rf, rf, logn);
    Zf(poly_neg)(rF, rF, logn);
    Zf(FFT)(rg, logn);
	Zf(FFT)(rG, logn);

	/*
	 * The Gram matrix is G = B·B*. Formulas are:
	 *   g00 = b00*adj(b00) + b01*adj(b01)
	 *   g01 = b00*adj(b10) + b01*adj(b11)
	 *   g10 = b10*adj(b00) + b11*adj(b01)
	 *   g11 = b10*adj(b10) + b11*adj(b11)
	 *
	 * For historical reasons, this implementation uses
	 * g00, g01 and g11 (upper triangle).
	 */
	g00 = (fpr *)tmp;
	g01 = g00 + n;
	g11 = g01 + n;
	gxx = g11 + n;

    Zf(poly_mulselfadj_fft)(g00, b00, logn);
    Zf(poly_mulselfadj_fft)(gxx, b01, logn);
    Zf(poly_add)(g00, g00, gxx, logn);

    Zf(poly_muladj_fft)(g01, b00, b10, logn);
    Zf(poly_muladj_fft)(gxx, b01, b11, logn);
    Zf(poly_add)(g01, g01, gxx, logn);

    Zf(poly_mulselfadj_fft)(g11, b10, logn);
    Zf(poly_mulselfadj_fft)(gxx, b11, logn);
    Zf(poly_add)(g11, g11, gxx, logn);

	/*
	 * Compute the Falcon tree.
	 */
	ffLDL_fft(tree, g00, g01, g11, logn, gxx);

	/*
	 * Normalize tree.
	 */
	ffLDL_binary_normalize(tree, logn, logn);
}

typedef int (*samplerZ)(void *ctx, fpr mu, fpr sigma);

/*
 * Perform Fast Fourier Sampling for target vector t. The Gram matrix
 * is provided (G = [[g00, g01], [adj(g01), g11]]). The sampled vector
 * is written over (t0,t1). The Gram matrix is modified as well. The
 * tmp[] buffer must have room for four polynomials.
 */
static void
ffSampling_fft_dyntree(samplerZ samp, void *samp_ctx,
	fpr *restrict t0, fpr *restrict t1,
	fpr *restrict g00, fpr *restrict g01, fpr *restrict g11,
	unsigned orig_logn, unsigned logn, fpr *restrict tmp)
{
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
	Zf(poly_LDL_fft)(g00, g01, g11, logn);

	/*
	 * Split d00 and d11 and expand them into half-size quasi-cyclic
	 * Gram matrices. We also save l10 in tmp[].
	 */
	Zf(poly_split_fft)(tmp, tmp + hn, g00, logn);
	memcpy(g00, tmp, n * sizeof *tmp);
	Zf(poly_split_fft)(tmp, tmp + hn, g11, logn);
	memcpy(g11, tmp, n * sizeof *tmp);
	memcpy(tmp, g01, n * sizeof *g01);
	memcpy(g01, g00, hn * sizeof *g00);
	memcpy(g01 + hn, g11, hn * sizeof *g00);

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
	Zf(poly_split_fft)(z1, z1 + hn, t1, logn);
	ffSampling_fft_dyntree(samp, samp_ctx, z1, z1 + hn,
		g11, g11 + hn, g01 + hn, orig_logn, logn - 1, z1 + n);
	Zf(poly_merge_fft)(tmp + (n << 1), z1, z1 + hn, logn);

	/*
	 * Compute tb0 = t0 + (t1 - z1) * l10.
	 * At that point, l10 is in tmp, t1 is unmodified, and z1 is
	 * in tmp + (n << 1). The buffer in z1 is free.
	 *
	 * In the end, z1 is written over t1, and tb0 is in t0.
	 */
	Zf(poly_sub)(z1, t1, tmp + (n << 1), logn);
	memcpy(t1, tmp + (n << 1), n * sizeof *tmp);
	Zf(poly_mul_fft)(tmp, tmp, z1, logn);
	Zf(poly_add)(t0, t0, tmp, logn);


	/*
	 * Second recursive invocation, on the split tb0 (currently in t0)
	 * and the left sub-tree.
	 */
	z0 = tmp;
	Zf(poly_split_fft)(z0, z0 + hn, t0, logn);
	ffSampling_fft_dyntree(samp, samp_ctx, z0, z0 + hn,
		g00, g00 + hn, g01, orig_logn, logn - 1, z0 + n);
	Zf(poly_merge_fft)(t0, z0, z0 + hn, logn);
}

/*
 * Perform Fast Fourier Sampling for target vector t and LDL tree T.
 * tmp[] must have size for at least two polynomials of size 2^logn.
 */
static void
ffSampling_fft(samplerZ samp, void *samp_ctx,
	fpr *restrict z0, fpr *restrict z1,
	const fpr *restrict tree,
	const fpr *restrict t0, const fpr *restrict t1, unsigned logn,
	fpr *restrict tmp)
{
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

        scvt = vdupq_n_s64(0);
        vload2(tmp, &t1[0]);
        a = tmp.val[0]; // a_re, a_im
        b = tmp.val[1]; // b_re, b_im
        vloadx2(tmp, &imagine[0]);
        neon_i21 = tmp.val[0];
        neon_1i2 = tmp.val[1];

        c = vaddq_f64(a, b);
        w01 = vmulq_n_f64(c, 0.5);

        c = vsubq_f64(a, b);
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
        w01 = vmulq_n_f64(w01, 0.5);

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
        scvt = vdupq_n_s64(0);
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
	 * We split t1 into z1 (reused as temporary storage), then do
	 * the recursive invocation, with output in tmp. We finally
	 * merge back into z1.
	 */
	Zf(poly_split_fft)(z1, z1 + hn, t1, logn);
	ffSampling_fft(samp, samp_ctx, tmp, tmp + hn,
		tree1, z1, z1 + hn, logn - 1, tmp + n);
	Zf(poly_merge_fft)(z1, tmp, tmp + hn, logn);

	/*
	 * Compute tb0 = t0 + (t1 - z1) * L. Value tb0 ends up in tmp[].
	 */
	Zf(poly_sub)(tmp, t1, z1, logn);
	Zf(poly_mul_fft)(tmp, tmp, tree, logn);
	Zf(poly_add)(tmp, tmp, t0, logn);

	/*
	 * Second recursive invocation.
	 */
	Zf(poly_split_fft)(z0, z0 + hn, tmp, logn);
	ffSampling_fft(samp, samp_ctx, tmp, tmp + hn,
		tree0, z0, z0 + hn, logn - 1, tmp + n);
	Zf(poly_merge_fft)(z0, tmp, tmp + hn, logn);
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
	const fpr *restrict expanded_key,
	const uint16_t *hm,
	unsigned logn, fpr *restrict tmp)
{
	size_t n;
	fpr *t0, *t1, *tx, *ty;
	const fpr *b00, *b01, *b10, *b11, *tree;
	fpr ni;
	uint32_t sqn;
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
    Zf(poly_fpr_of_s16)(t0, hm, n);

	/*
	 * Apply the lattice basis to obtain the real target
	 * vector (after normalization with regards to modulus).
	 */
	Zf(FFT)(t0, logn);
	ni = fpr_inverse_of_q;
	Zf(poly_mul_fft)(t1, t0, b01, logn);
	Zf(poly_mulconst)(t1, t1, fpr_neg(ni), logn);
	Zf(poly_mul_fft)(t0, t0, b11, logn);
	Zf(poly_mulconst)(t0, t0, ni, logn);

	tx = t1 + n;
	ty = tx + n;

	/*
	 * Apply sampling. Output is written back in [tx, ty].
	 */
	ffSampling_fft(samp, samp_ctx, tx, ty, tree, t0, t1, logn, ty + n);

	/*
	 * Get the lattice point corresponding to that tiny vector.
	 */
	memcpy(t0, tx, n * sizeof *tx);
	memcpy(t1, ty, n * sizeof *ty);
	Zf(poly_mul_fft)(tx, tx, b00, logn);
	Zf(poly_mul_fft)(ty, ty, b10, logn);
	Zf(poly_add)(tx, tx, ty, logn);
	Zf(poly_mul_fft)(ty, t0, b01, logn);
    
	memcpy(t0, tx, n * sizeof *tx);
	Zf(poly_mul_fft)(t1, t1, b11, logn);
	Zf(poly_add)(t1, t1, ty, logn);

	Zf(iFFT)(t0, logn);
	Zf(iFFT)(t1, logn);

	/*
	 * Compute the signature.
	 */

	/*
	 * With "normal" degrees (e.g. 512 or 1024), it is very
	 * improbable that the computed vector is not short enough;
	 * however, it may happen in practice for the very reduced
	 * versions (e.g. degree 16 or below). In that case, the caller
	 * will loop, and we must not write anything into s2[] because
	 * s2[] may overlap with the hashed message hm[] and we need
	 * hm[] for the next iteration.
	 */

    s1tmp = (int16_t *)tx;
	s2tmp = (int16_t *)tmp;
	
    Zf(sign_short_s1)(&sqn, s1tmp, hm, t0, n);
    Zf(sign_short_s2)(s2tmp, t1, n);


	if (Zf(is_short_half)(sqn, s2tmp, logn)) {
		memcpy(s2, s2tmp, n * sizeof *s2);
		memcpy(tmp, s1tmp, n * sizeof *s1tmp);
		return 1;
	}
	return 0;
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
	const int8_t *restrict f, const int8_t *restrict g,
	const int8_t *restrict F, const int8_t *restrict G,
	const uint16_t *hm, unsigned logn, fpr *restrict tmp)
{
	fpr *t0, *t1, *tx, *ty;
	fpr *b00, *b01, *b10, *b11, *g00, *g01, *g11;
	fpr ni;
	uint32_t sqn;
	int16_t *s1tmp, *s2tmp;

	const unsigned n = MKN(logn);

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
    Zf(FFT)(b01, logn);
	Zf(FFT)(b11, logn);
    Zf(poly_neg)(b01, b01, logn);
    Zf(poly_neg)(b11, b11, logn);
    Zf(FFT)(b00, logn);
	Zf(FFT)(b10, logn);

	/*
	 * Compute the Gram matrix G = B·B*. Formulas are:
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

	Zf(poly_mulselfadj_fft)(t0, b01, logn);    // t0 <- b01*adj(b01)

	Zf(poly_muladj_fft)(t1, b00, b10, logn);   // t1 <- b00*adj(b10)
	Zf(poly_mulselfadj_fft)(b00, b00, logn);   // b00 <- b00*adj(b00)
	Zf(poly_add)(b00, b00, t0, logn);      // b00 <- g00
	
	memcpy(t0, b01, n * sizeof *b01);
    Zf(poly_muladj_fft)(b01, b01, b11, logn);  // b01 <- b01*adj(b11)
	Zf(poly_add)(b01, b01, t1, logn);      // b01 <- g01
	
    Zf(poly_mulselfadj_fft)(b10, b10, logn);   // b10 <- b10*adj(b10)
	Zf(poly_mulselfadj_fft)(t1, b11, logn);    // t1 <- b11*adj(b11)
	Zf(poly_add)(b10, b10, t1, logn);      // b10 <- g11

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
    Zf(poly_fpr_of_s16)(t0, hm, n);

    
	/*
	 * Apply the lattice basis to obtain the real target
	 * vector (after normalization with regards to modulus).
	 */
	Zf(FFT)(t0, logn);
	ni = fpr_inverse_of_q;
	Zf(poly_mul_fft)(t1, t0, b01, logn);
	Zf(poly_mulconst)(t1, t1, fpr_neg(ni), logn);
	Zf(poly_mul_fft)(t0, t0, b11, logn);
	Zf(poly_mulconst)(t0, t0, ni, logn);
  
	/*
	 * b01 and b11 can be discarded, so we move back (t0,t1).
	 * Memory layout is now:
	 *      g00 g01 g11 t0 t1
	 */
	memcpy(b11, t0, n * 2 * sizeof *t0);
	t0 = g11 + n;
	t1 = t0 + n;

	/*
	 * Apply sampling; result is written over (t0,t1).
     * t1, g00
	 */
	ffSampling_fft_dyntree(samp, samp_ctx,
		t0, t1, g00, g01, g11, logn, logn, t1 + n);
    
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
	memmove(b11 + n, t0, n * 2 * sizeof *t0);
	t0 = b11 + n;
	t1 = t0 + n;
	smallints_to_fpr(b01, f, logn);
	smallints_to_fpr(b00, g, logn);
	smallints_to_fpr(b11, F, logn);
	smallints_to_fpr(b10, G, logn);
	Zf(FFT)(b01, logn);
	Zf(FFT)(b11, logn);
    Zf(poly_neg)(b01, b01, logn);
    Zf(poly_neg)(b11, b11, logn);
	Zf(FFT)(b00, logn);
	Zf(FFT)(b10, logn);
	tx = t1 + n;
	ty = tx + n;

	/*
	 * Get the lattice point corresponding to that tiny vector.
	 */


	Zf(poly_mul_fft)(tx, t0, b00, logn);
	Zf(poly_mul_fft)(ty, t1, b10, logn);
	Zf(poly_add)(tx, tx, ty, logn);
	Zf(poly_mul_fft)(ty, t0, b01, logn);

	memcpy(t0, tx, n * sizeof *tx);
	Zf(poly_mul_fft)(t1, t1, b11, logn);
	Zf(poly_add)(t1, t1, ty, logn);
	Zf(iFFT)(t0, logn);
	Zf(iFFT)(t1, logn);


	/*
	 * With "normal" degrees (e.g. 512 or 1024), it is very
	 * improbable that the computed vector is not short enough;
	 * however, it may happen in practice for the very reduced
	 * versions (e.g. degree 16 or below). In that case, the caller
	 * will loop, and we must not write anything into s2[] because
	 * s2[] may overlap with the hashed message hm[] and we need
	 * hm[] for the next iteration.
	 */
	s1tmp = (int16_t *)tx;
	s2tmp = (int16_t *)tmp;
	
    Zf(sign_short_s1)(&sqn, s1tmp, hm, t0, n);
    Zf(sign_short_s2)(s2tmp, t1, n);

	if (Zf(is_short_half)(sqn, s2tmp, logn)) {
		memcpy(s2, s2tmp, n * sizeof *s2);
		memcpy(tmp, s1tmp, n * sizeof *s1tmp);
		return 1;
	}
	return 0;
}


/* see inner.h */
void
Zf(sign_tree)(int16_t *sig, inner_shake256_context *rng,
	const fpr *restrict expanded_key,
	const uint16_t *hm, unsigned logn, uint8_t *tmp)
{
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
		spc.sigma_min = fpr_sigma_min_9;
		Zf(prng_init)(&spc.p, rng);
		samp = Zf(sampler);
		samp_ctx = &spc;

		/*
		 * Do the actual signature.
		 */
		if (do_sign_tree(samp, samp_ctx, sig,
			expanded_key, hm, logn, ftmp))
		{
			break;
		}
	}
}

/* see inner.h */
void
Zf(sign_dyn)(int16_t *sig, inner_shake256_context *rng,
	const int8_t *restrict f, const int8_t *restrict g,
	const int8_t *restrict F, const int8_t *restrict G,
	const uint16_t *hm, unsigned logn, uint8_t *tmp)
{
	fpr *ftmp;

	ftmp = (fpr *)tmp;
    int i =0;
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
        // TODO: hardcode this
        if (logn == 9)
        {
            spc.sigma_min = fpr_sigma_min_9;
        }
        else if (logn == 10)
        {
            spc.sigma_min = fpr_sigma_min_10;
        }
		Zf(prng_init)(&spc.p, rng);
		samp = Zf(sampler);
		samp_ctx = &spc;

		/*
		 * Do the actual signature.
		 */
		if (do_sign_dyn(samp, samp_ctx, sig,
			f, g, F, G, hm, logn, ftmp))
		{
			break;
		}
        printf("%d, ", ++i);
	}
}
