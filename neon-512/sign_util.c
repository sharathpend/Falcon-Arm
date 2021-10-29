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
ffLDL_treesize(unsigned logn) {
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
ffLDL_fft_inner(fpr *tree,
                fpr *g0, fpr *g1, unsigned logn, fpr *tmp) {
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
    PQCLEAN_FALCON512_NEON_poly_LDLmv_fft(tmp, tree, g0, g1, g0, logn);

    /*
     * Split d00 (currently in g0) and d11 (currently in tmp). We
     * reuse g0 and g1 as temporary storage spaces:
     *   d00 splits into g1, g1+hn
     *   d11 splits into g0, g0+hn
     */
    PQCLEAN_FALCON512_NEON_poly_split_fft(g1, g1 + hn, g0, logn);
    PQCLEAN_FALCON512_NEON_poly_split_fft(g0, g0 + hn, tmp, logn);

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
ffLDL_fft(fpr *tree, const fpr *g00,
          const fpr *g01, const fpr *g11,
          unsigned logn, fpr *tmp) {
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

    memcpy(d00, g00, n * sizeof * g00);
    PQCLEAN_FALCON512_NEON_poly_LDLmv_fft(d11, tree, g00, g01, g11, logn);

    PQCLEAN_FALCON512_NEON_poly_split_fft(tmp, tmp + hn, d00, logn);
    PQCLEAN_FALCON512_NEON_poly_split_fft(d00, d00 + hn, d11, logn);
    memcpy(d11, tmp, n * sizeof * tmp);
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
ffLDL_binary_normalize(fpr *tree, unsigned logn) {
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
        ffLDL_binary_normalize(tree + n, logn - 1);
        ffLDL_binary_normalize(tree + n + ffLDL_treesize(logn - 1),
                               logn - 1);
    }
}

/* =================================================================== */

/*
 * Convert an integer polynomial (with small values) into the
 * representation with complex numbers.
 */
static void
smallints_to_fpr(fpr *r, const int8_t *t, unsigned logn) {
    size_t n, u;

    n = MKN(logn);
    for (u = 0; u < n; u ++) {
        r[u] = fpr_of(t[u]);
    }
}

/*
 * The expanded private key contains:
 *  - The B0 matrix (four elements)
 *  - The ffLDL tree
 */

static inline size_t
skoff_b00(unsigned logn) {
    (void)logn;
    return 0;
}

static inline size_t
skoff_b01(unsigned logn) {
    return MKN(logn);
}

static inline size_t
skoff_b10(unsigned logn) {
    return 2 * MKN(logn);
}

static inline size_t
skoff_b11(unsigned logn) {
    return 3 * MKN(logn);
}

static inline size_t
skoff_tree(unsigned logn) {
    return 4 * MKN(logn);
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_expand_privkey(fpr *expanded_key,
                                       const int8_t *f, const int8_t *g,
                                       const int8_t *F, const int8_t *G,
                                       unsigned logn, uint8_t *tmp) {
    size_t n;
    fpr *rf, *rg, *rF, *rG;
    fpr *b00, *b01, *b10, *b11;
    fpr *g00, *g01, *g11, *gxx;
    fpr *tree;

    n = MKN(logn);
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
    PQCLEAN_FALCON512_NEON_FFT(rf, logn);
    PQCLEAN_FALCON512_NEON_FFT(rg, logn);
    PQCLEAN_FALCON512_NEON_FFT(rF, logn);
    PQCLEAN_FALCON512_NEON_FFT(rG, logn);
    PQCLEAN_FALCON512_NEON_poly_neg(rf, logn);
    PQCLEAN_FALCON512_NEON_poly_neg(rF, logn);

    /*
     * The Gram matrix is G = B x B*. Formulas are:
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

    memcpy(g00, b00, n * sizeof * b00);
    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(g00, logn);
    memcpy(gxx, b01, n * sizeof * b01);
    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(gxx, logn);
    PQCLEAN_FALCON512_NEON_poly_add(g00, gxx, logn);

    memcpy(g01, b00, n * sizeof * b00);
    PQCLEAN_FALCON512_NEON_poly_muladj_fft(g01, b10, logn);
    memcpy(gxx, b01, n * sizeof * b01);
    PQCLEAN_FALCON512_NEON_poly_muladj_fft(gxx, b11, logn);
    PQCLEAN_FALCON512_NEON_poly_add(g01, gxx, logn);

    memcpy(g11, b10, n * sizeof * b10);
    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(g11, logn);
    memcpy(gxx, b11, n * sizeof * b11);
    PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(gxx, logn);
    PQCLEAN_FALCON512_NEON_poly_add(g11, gxx, logn);

    /*
     * Compute the Falcon tree.
     */
    ffLDL_fft(tree, g00, g01, g11, logn, gxx);

    /*
     * Normalize tree.
     */
    ffLDL_binary_normalize(tree, logn);
}
