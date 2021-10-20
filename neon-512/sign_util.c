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
void
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

/*
 * Sample an integer value along a half-gaussian distribution centered
 * on zero and standard deviation 1.8205, with a precision of 72 bits.
 */
int
PQCLEAN_FALCON512_NEON_gaussian0_sampler(prng *p) {

    static const uint32_t dist[] = {
        10745844u,  3068844u,  3741698u,
        5559083u,  1580863u,  8248194u,
        2260429u, 13669192u,  2736639u,
        708981u,  4421575u, 10046180u,
        169348u,  7122675u,  4136815u,
        30538u, 13063405u,  7650655u,
        4132u, 14505003u,  7826148u,
        417u, 16768101u, 11363290u,
        31u,  8444042u,  8086568u,
        1u, 12844466u,   265321u,
        0u,  1232676u, 13644283u,
        0u,    38047u,  9111839u,
        0u,      870u,  6138264u,
        0u,       14u, 12545723u,
        0u,        0u,  3104126u,
        0u,        0u,    28824u,
        0u,        0u,      198u,
        0u,        0u,        1u
    };

    uint32_t v0, v1, v2, hi;
    uint64_t lo;
    size_t u;
    int z;

    /*
     * Get a random 72-bit value, into three 24-bit limbs v0..v2.
     */
    lo = prng_get_u64(p);
    hi = prng_get_u8(p);
    v0 = (uint32_t)lo & 0xFFFFFF;
    v1 = (uint32_t)(lo >> 24) & 0xFFFFFF;
    v2 = (uint32_t)(lo >> 48) | (hi << 16);

    /*
     * Sampled value is z, such that v0..v2 is lower than the first
     * z elements of the table.
     */
    z = 0;
    for (u = 0; u < (sizeof dist) / sizeof(dist[0]); u += 3) {
        uint32_t w0, w1, w2, cc;

        w0 = dist[u + 2];
        w1 = dist[u + 1];
        w2 = dist[u + 0];
        cc = (v0 - w0) >> 31;
        cc = (v1 - w1 - cc) >> 31;
        cc = (v2 - w2 - cc) >> 31;
        z += (int)cc;
    }
    return z;

}

/*
 * Sample a bit with probability exp(-x) for some x >= 0.
 */
static int
BerExp(prng *p, fpr x, fpr ccs) {
    int s, i;
    fpr r;
    uint32_t sw, w;
    uint64_t z;

    /*
     * Reduce x modulo log(2): x = s*log(2) + r, with s an integer,
     * and 0 <= r < log(2). Since x >= 0, we can use fpr_trunc().
     */
    s = (int)fpr_trunc(fpr_mul(x, fpr_inv_log2));
    r = fpr_sub(x, fpr_mul(fpr_of(s), fpr_log2));

    /*
     * It may happen (quite rarely) that s >= 64; if sigma = 1.2
     * (the minimum value for sigma), r = 0 and b = 1, then we get
     * s >= 64 if the half-Gaussian produced a z >= 13, which happens
     * with probability about 0.000000000230383991, which is
     * approximatively equal to 2^(-32). In any case, if s >= 64,
     * then BerExp will be non-zero with probability less than
     * 2^(-64), so we can simply saturate s at 63.
     */
    sw = (uint32_t)s;
    sw ^= (sw ^ 63) & -((63 - sw) >> 31);
    s = (int)sw;

    /*
     * Compute exp(-r); we know that 0 <= r < log(2) at this point, so
     * we can use fpr_expm_p63(), which yields a result scaled to 2^63.
     * We scale it up to 2^64, then right-shift it by s bits because
     * we really want exp(-x) = 2^(-s)*exp(-r).
     *
     * The "-1" operation makes sure that the value fits on 64 bits
     * (i.e. if r = 0, we may get 2^64, and we prefer 2^64-1 in that
     * case). The bias is negligible since fpr_expm_p63() only computes
     * with 51 bits of precision or so.
     */
    z = ((fpr_expm_p63(r, ccs) << 1) - 1) >> s;

    /*
     * Sample a bit with probability exp(-x). Since x = s*log(2) + r,
     * exp(-x) = 2^-s * exp(-r), we compare lazily exp(-x) with the
     * PRNG output to limit its consumption, the sign of the difference
     * yields the expected result.
     */
    i = 64;
    do {
        i -= 8;
        w = prng_get_u8(p) - ((uint32_t)(z >> i) & 0xFF);
    } while (!w && i > 0);
    return (int)(w >> 31);
}

/*
 * The sampler produces a random integer that follows a discrete Gaussian
 * distribution, centered on mu, and with standard deviation sigma. The
 * provided parameter isigma is equal to 1/sigma.
 *
 * The value of sigma MUST lie between 1 and 2 (i.e. isigma lies between
 * 0.5 and 1); in Falcon, sigma should always be between 1.2 and 1.9.
 */
int
PQCLEAN_FALCON512_NEON_sampler(void *ctx, fpr mu, fpr isigma) {
    sampler_context *spc;
    int s, z0, z, b;
    fpr r, dss, ccs, x;

    spc = ctx;

    /*
     * Center is mu. We compute mu = s + r where s is an integer
     * and 0 <= r < 1.
     */
    s = (int)fpr_floor(mu);
    r = fpr_sub(mu, fpr_of(s));

    /*
     * dss = 1/(2*sigma^2) = 0.5*(isigma^2).
     */
    dss = fpr_half(fpr_sqr(isigma));

    /*
     * ccs = sigma_min / sigma = sigma_min * isigma.
     */
    ccs = fpr_mul(isigma, spc->sigma_min);

    /*
     * We now need to sample on center r.
     */
    for (;;) {
        /*
         * Sample z for a Gaussian distribution. Then get a
         * random bit b to turn the sampling into a bimodal
         * distribution: if b = 1, we use z+1, otherwise we
         * use -z. We thus have two situations:
         *
         *  - b = 1: z >= 1 and sampled against a Gaussian
         *    centered on 1.
         *  - b = 0: z <= 0 and sampled against a Gaussian
         *    centered on 0.
         */
        z0 = PQCLEAN_FALCON512_NEON_gaussian0_sampler(&spc->p);
        b = (int)prng_get_u8(&spc->p) & 1;
        z = b + ((b << 1) - 1) * z0;

        /*
         * Rejection sampling. We want a Gaussian centered on r;
         * but we sampled against a Gaussian centered on b (0 or
         * 1). But we know that z is always in the range where
         * our sampling distribution is greater than the Gaussian
         * distribution, so rejection works.
         *
         * We got z with distribution:
         *    G(z) = exp(-((z-b)^2)/(2*sigma0^2))
         * We target distribution:
         *    S(z) = exp(-((z-r)^2)/(2*sigma^2))
         * Rejection sampling works by keeping the value z with
         * probability S(z)/G(z), and starting again otherwise.
         * This requires S(z) <= G(z), which is the case here.
         * Thus, we simply need to keep our z with probability:
         *    P = exp(-x)
         * where:
         *    x = ((z-r)^2)/(2*sigma^2) - ((z-b)^2)/(2*sigma0^2)
         *
         * Here, we scale up the Bernouilli distribution, which
         * makes rejection more probable, but makes rejection
         * rate sufficiently decorrelated from the Gaussian
         * center and standard deviation that the whole sampler
         * can be said to be constant-time.
         */
        x = fpr_mul(fpr_sqr(fpr_sub(fpr_of(z), r)), dss);
        x = fpr_sub(x, fpr_mul(fpr_of(z0 * z0), fpr_inv_2sqrsigma0));
        if (BerExp(&spc->p, x, ccs)) {
            /*
             * Rejection sampling was centered on r, but the
             * actual center is mu = s + r.
             */
            return s + z;
        }
    }
}
