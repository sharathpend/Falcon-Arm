#include "inner.h"

/* 
 * c = (a * b)* scalar_x
 */
void PQCLEAN_FALCON512_NEON_poly_mul_fftconst(fpr *c, const fpr *a, const fpr *b, const fpr x)
{
    // Total 32 registers
    float64x2x4_t a_re, b_re, a_im, b_im, tmp1, tmp2; // 24
    float64x2x4_t d_re, d_im;                         // 8
    float64x2_t neon_x;
    neon_x = vdupq_n_f64(x);

    for (int i = 0; i < FALCON_N / 2; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(b_re, &b[i]);
        vloadx4(a_im, &a[i + FALCON_N / 2]);
        vloadx4(b_im, &b[i + FALCON_N / 2]);

        vfmul(tmp1.val[0], a_re.val[0], b_re.val[0]);
        vfmul(tmp1.val[1], a_re.val[1], b_re.val[1]);
        vfmul(tmp1.val[2], a_re.val[2], b_re.val[2]);
        vfmul(tmp1.val[3], a_re.val[3], b_re.val[3]);

        vfms(d_re.val[0], tmp1.val[0], a_im.val[0], b_im.val[0]);
        vfms(d_re.val[1], tmp1.val[1], a_im.val[1], b_im.val[1]);
        vfms(d_re.val[2], tmp1.val[2], a_im.val[2], b_im.val[2]);
        vfms(d_re.val[3], tmp1.val[3], a_im.val[3], b_im.val[3]);

        vfmul(d_re.val[0], d_re.val[0], neon_x);
        vfmul(d_re.val[1], d_re.val[1], neon_x);
        vfmul(d_re.val[2], d_re.val[2], neon_x);
        vfmul(d_re.val[3], d_re.val[3], neon_x);

        vfmul(tmp2.val[0], a_re.val[0], b_im.val[0]);
        vfmul(tmp2.val[1], a_re.val[1], b_im.val[1]);
        vfmul(tmp2.val[2], a_re.val[2], b_im.val[2]);
        vfmul(tmp2.val[3], a_re.val[3], b_im.val[3]);

        vfma(d_im.val[0], tmp2.val[0], a_im.val[0], b_re.val[0]);
        vfma(d_im.val[1], tmp2.val[1], a_im.val[1], b_re.val[1]);
        vfma(d_im.val[2], tmp2.val[2], a_im.val[2], b_re.val[2]);
        vfma(d_im.val[3], tmp2.val[3], a_im.val[3], b_re.val[3]);

        vfmul(d_im.val[0], d_im.val[0], neon_x);
        vfmul(d_im.val[1], d_im.val[1], neon_x);
        vfmul(d_im.val[2], d_im.val[2], neon_x);
        vfmul(d_im.val[3], d_im.val[3], neon_x);

        vstorex4(&c[i], d_re);
        vstorex4(&c[i + FALCON_N / 2], d_im);
    }
}

/* 
 * c = a * b
 */
void PQCLEAN_FALCON512_NEON_poly_mul_fft(fpr *c, const fpr *a, const fpr *b)
{
    // Total 32 registers
    float64x2x4_t a_re, b_re, a_im, b_im, tmp1, tmp2; // 24
    float64x2x4_t d_re, d_im;                         // 8

    for (int i = 0; i < FALCON_N / 2; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(b_re, &b[i]);
        vloadx4(a_im, &a[i + FALCON_N / 2]);
        vloadx4(b_im, &b[i + FALCON_N / 2]);

        vfmul(tmp1.val[0], a_re.val[0], b_re.val[0]);
        vfmul(tmp1.val[1], a_re.val[1], b_re.val[1]);
        vfmul(tmp1.val[2], a_re.val[2], b_re.val[2]);
        vfmul(tmp1.val[3], a_re.val[3], b_re.val[3]);

        vfms(d_re.val[0], tmp1.val[0], a_im.val[0], b_im.val[0]);
        vfms(d_re.val[1], tmp1.val[1], a_im.val[1], b_im.val[1]);
        vfms(d_re.val[2], tmp1.val[2], a_im.val[2], b_im.val[2]);
        vfms(d_re.val[3], tmp1.val[3], a_im.val[3], b_im.val[3]);

        vfmul(tmp2.val[0], a_re.val[0], b_im.val[0]);
        vfmul(tmp2.val[1], a_re.val[1], b_im.val[1]);
        vfmul(tmp2.val[2], a_re.val[2], b_im.val[2]);
        vfmul(tmp2.val[3], a_re.val[3], b_im.val[3]);

        vfma(d_im.val[0], tmp2.val[0], a_im.val[0], b_re.val[0]);
        vfma(d_im.val[1], tmp2.val[1], a_im.val[1], b_re.val[1]);
        vfma(d_im.val[2], tmp2.val[2], a_im.val[2], b_re.val[2]);
        vfma(d_im.val[3], tmp2.val[3], a_im.val[3], b_re.val[3]);

        vstorex4(&c[i], d_re);
        vstorex4(&c[i + FALCON_N / 2], d_im);
    }
}

/* 
 * c = a * scalar_x
 */
void PQCLEAN_FALCON512_NEON_poly_mulconst(fpr *c, const fpr *a, const fpr x)
{
    // Total 9 registers
    float64x2x4_t neon_a, neon_c;
    float64x2_t neon_x;
    neon_x = vdupq_n_f64(x);
    for (int i = 0; i < FALCON_N; i += 8)
    {
        vloadx4(neon_a, &a[i]);
        vfmul(neon_c.val[0], neon_a.val[0], neon_x);
        vfmul(neon_c.val[1], neon_a.val[1], neon_x);
        vfmul(neon_c.val[2], neon_a.val[2], neon_x);
        vfmul(neon_c.val[3], neon_a.val[3], neon_x);
        vstorex4(&c[i], neon_c);
    }
}

/* 
 * d = c + a *b
 */
void PQCLEAN_FALCON512_NEON_poly_mul_fft_add(fpr *d, const fpr *c, const fpr *a, const fpr *b)
{
    // Total 32 registers
    float64x2x4_t a_re, b_re, a_im, b_im, c_re, c_im, tmp; // 24
    float64x2x4_t d_re, d_im;                              // 8

    for (int i = 0; i < FALCON_N / 2; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(b_re, &b[i]);
        vloadx4(c_re, &c[i]);
        vloadx4(a_im, &a[i + FALCON_N / 2]);
        vloadx4(b_im, &b[i + FALCON_N / 2]);
        vloadx4(c_im, &c[i + FALCON_N / 2]);

        vfmul(tmp.val[0], a_re.val[0], b_re.val[0]);
        vfmul(tmp.val[1], a_re.val[1], b_re.val[1]);
        vfmul(tmp.val[2], a_re.val[2], b_re.val[2]);
        vfmul(tmp.val[3], a_re.val[3], b_re.val[3]);

        vfms(d_re.val[0], tmp.val[0], a_im.val[0], b_im.val[0]);
        vfms(d_re.val[1], tmp.val[1], a_im.val[1], b_im.val[1]);
        vfms(d_re.val[2], tmp.val[2], a_im.val[2], b_im.val[2]);
        vfms(d_re.val[3], tmp.val[3], a_im.val[3], b_im.val[3]);

        vfmul(tmp.val[0], a_re.val[0], b_im.val[0]);
        vfmul(tmp.val[1], a_re.val[1], b_im.val[1]);
        vfmul(tmp.val[2], a_re.val[2], b_im.val[2]);
        vfmul(tmp.val[3], a_re.val[3], b_im.val[3]);

        vfma(d_im.val[0], tmp.val[0], a_im.val[0], b_re.val[0]);
        vfma(d_im.val[1], tmp.val[1], a_im.val[1], b_re.val[1]);
        vfma(d_im.val[2], tmp.val[2], a_im.val[2], b_re.val[2]);
        vfma(d_im.val[3], tmp.val[3], a_im.val[3], b_re.val[3]);

        vfadd(d_re.val[0], d_re.val[0], c_re.val[0]);
        vfadd(d_re.val[1], d_re.val[1], c_re.val[1]);
        vfadd(d_re.val[2], d_re.val[2], c_re.val[2]);
        vfadd(d_re.val[3], d_re.val[3], c_re.val[3]);

        vfadd(d_im.val[0], d_im.val[0], c_im.val[0]);
        vfadd(d_im.val[1], d_im.val[1], c_im.val[1]);
        vfadd(d_im.val[2], d_im.val[2], c_im.val[2]);
        vfadd(d_im.val[3], d_im.val[3], c_im.val[3]);

        vstorex4(&c[i], d_re);
        vstorex4(&c[i + FALCON_N / 2], d_im);
    }
}

/* 
 * c = a + b
 */
void PQCLEAN_FALCON512_NEON_poly_add(fpr *c, const fpr *a, const fpr *b)
{
    float64x2x4_t neon_a, neon_b, neon_c;
    for (int i = 0; i < FALCON_N; i += 8)
    {
        vloadx4(neon_a, &a[i]);
        vloadx4(neon_b, &b[i]);

        vfadd(neon_c.val[0], neon_a.val[0], neon_b.val[0]);
        vfadd(neon_c.val[1], neon_a.val[1], neon_b.val[1]);
        vfadd(neon_c.val[2], neon_a.val[2], neon_b.val[2]);
        vfadd(neon_c.val[3], neon_a.val[3], neon_b.val[3]);

        vstorex4(&c[i], neon_c);
    }
}

/* 
 * c = a - b
 */
void PQCLEAN_FALCON512_NEON_poly_sub(fpr *c, const fpr *a, const fpr *b)
{
    float64x2x4_t neon_a, neon_b, neon_c;
    for (int i = 0; i < FALCON_N; i += 8)
    {
        vloadx4(neon_a, &a[i]);
        vloadx4(neon_b, &b[i]);

        vfsub(neon_c.val[0], neon_a.val[0], neon_b.val[0]);
        vfsub(neon_c.val[1], neon_a.val[1], neon_b.val[1]);
        vfsub(neon_c.val[2], neon_a.val[2], neon_b.val[2]);
        vfsub(neon_c.val[3], neon_a.val[3], neon_b.val[3]);

        vstorex4(&c[i], neon_c);
    }
}


/////////////
// TODO: vectorize the code below
/*
 * FFT code.
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


/*
 * Rules for complex number macros:
 * --------------------------------
 *
 * Operand order is: destination, source1, source2...
 *
 * Each operand is a real and an imaginary part.
 *
 * All overlaps are allowed.
 */

/*
 * Addition of two complex numbers (d = a + b).
 */
#define FPC_ADD(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_re, fpct_im; \
        fpct_re = fpr_add(a_re, b_re); \
        fpct_im = fpr_add(a_im, b_im); \
        (d_re) = fpct_re; \
        (d_im) = fpct_im; \
    } while (0)

/*
 * Subtraction of two complex numbers (d = a - b).
 */
#define FPC_SUB(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_re, fpct_im; \
        fpct_re = fpr_sub(a_re, b_re); \
        fpct_im = fpr_sub(a_im, b_im); \
        (d_re) = fpct_re; \
        (d_im) = fpct_im; \
    } while (0)

/*
 * Multplication of two complex numbers (d = a * b).
 */
#define FPC_MUL(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_a_re, fpct_a_im; \
        fpr fpct_b_re, fpct_b_im; \
        fpr fpct_d_re, fpct_d_im; \
        fpct_a_re = (a_re); \
        fpct_a_im = (a_im); \
        fpct_b_re = (b_re); \
        fpct_b_im = (b_im); \
        fpct_d_re = fpr_sub( \
                             fpr_mul(fpct_a_re, fpct_b_re), \
                             fpr_mul(fpct_a_im, fpct_b_im)); \
        fpct_d_im = fpr_add( \
                             fpr_mul(fpct_a_re, fpct_b_im), \
                             fpr_mul(fpct_a_im, fpct_b_re)); \
        (d_re) = fpct_d_re; \
        (d_im) = fpct_d_im; \
    } while (0)

/*
 * Squaring of a complex number (d = a * a).
 */
#define FPC_SQR(d_re, d_im, a_re, a_im)   do { \
        fpr fpct_a_re, fpct_a_im; \
        fpr fpct_d_re, fpct_d_im; \
        fpct_a_re = (a_re); \
        fpct_a_im = (a_im); \
        fpct_d_re = fpr_sub(fpr_sqr(fpct_a_re), fpr_sqr(fpct_a_im)); \
        fpct_d_im = fpr_double(fpr_mul(fpct_a_re, fpct_a_im)); \
        (d_re) = fpct_d_re; \
        (d_im) = fpct_d_im; \
    } while (0)

/*
 * Inversion of a complex number (d = 1 / a).
 */
#define FPC_INV(d_re, d_im, a_re, a_im)   do { \
        fpr fpct_a_re, fpct_a_im; \
        fpr fpct_d_re, fpct_d_im; \
        fpr fpct_m; \
        fpct_a_re = (a_re); \
        fpct_a_im = (a_im); \
        fpct_m = fpr_add(fpr_sqr(fpct_a_re), fpr_sqr(fpct_a_im)); \
        fpct_m = fpr_inv(fpct_m); \
        fpct_d_re = fpr_mul(fpct_a_re, fpct_m); \
        fpct_d_im = fpr_mul(fpr_neg(fpct_a_im), fpct_m); \
        (d_re) = fpct_d_re; \
        (d_im) = fpct_d_im; \
    } while (0)

/*
 * Division of complex numbers (d = a / b).
 */
#define FPC_DIV(d_re, d_im, a_re, a_im, b_re, b_im)   do { \
        fpr fpct_a_re, fpct_a_im; \
        fpr fpct_b_re, fpct_b_im; \
        fpr fpct_d_re, fpct_d_im; \
        fpr fpct_m; \
        fpct_a_re = (a_re); \
        fpct_a_im = (a_im); \
        fpct_b_re = (b_re); \
        fpct_b_im = (b_im); \
        fpct_m = fpr_add(fpr_sqr(fpct_b_re), fpr_sqr(fpct_b_im)); \
        fpct_m = fpr_inv(fpct_m); \
        fpct_b_re = fpr_mul(fpct_b_re, fpct_m); \
        fpct_b_im = fpr_mul(fpr_neg(fpct_b_im), fpct_m); \
        fpct_d_re = fpr_sub( \
                             fpr_mul(fpct_a_re, fpct_b_re), \
                             fpr_mul(fpct_a_im, fpct_b_im)); \
        fpct_d_im = fpr_add( \
                             fpr_mul(fpct_a_re, fpct_b_im), \
                             fpr_mul(fpct_a_im, fpct_b_re)); \
        (d_re) = fpct_d_re; \
        (d_im) = fpct_d_im; \
    } while (0)


/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_neg(fpr *a, unsigned logn) {
    size_t n, u;

    n = (size_t)1 << logn;
    for (u = 0; u < n; u ++) {
        a[u] = fpr_neg(a[u]);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_adj_fft(fpr *a, unsigned logn) {
    size_t n, u;

    n = (size_t)1 << logn;
    for (u = (n >> 1); u < n; u ++) {
        a[u] = fpr_neg(a[u]);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_mul_fft(
    fpr *a, const fpr *b, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr a_re, a_im, b_re, b_im;

        a_re = a[u];
        a_im = a[u + hn];
        b_re = b[u];
        b_im = b[u + hn];
        FPC_MUL(a[u], a[u + hn], a_re, a_im, b_re, b_im);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_muladj_fft(
    fpr *a, const fpr *b, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr a_re, a_im, b_re, b_im;

        a_re = a[u];
        a_im = a[u + hn];
        b_re = b[u];
        b_im = fpr_neg(b[u + hn]);
        FPC_MUL(a[u], a[u + hn], a_re, a_im, b_re, b_im);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_mulselfadj_fft(fpr *a, unsigned logn) {
    /*
     * Since each coefficient is multiplied with its own conjugate,
     * the result contains only real values.
     */
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr a_re, a_im;

        a_re = a[u];
        a_im = a[u + hn];
        a[u] = fpr_add(fpr_sqr(a_re), fpr_sqr(a_im));
        a[u + hn] = fpr_zero;
    }
}


/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_div_fft(
    fpr *a, const fpr *b, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr a_re, a_im, b_re, b_im;

        a_re = a[u];
        a_im = a[u + hn];
        b_re = b[u];
        b_im = b[u + hn];
        FPC_DIV(a[u], a[u + hn], a_re, a_im, b_re, b_im);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_invnorm2_fft(fpr *d,
        const fpr *a, const fpr *b, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr a_re, a_im;
        fpr b_re, b_im;

        a_re = a[u];
        a_im = a[u + hn];
        b_re = b[u];
        b_im = b[u + hn];
        d[u] = fpr_inv(fpr_add(
                           fpr_add(fpr_sqr(a_re), fpr_sqr(a_im)),
                           fpr_add(fpr_sqr(b_re), fpr_sqr(b_im))));
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_add_muladj_fft(fpr *d,
        const fpr *F, const fpr *G,
        const fpr *f, const fpr *g, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr F_re, F_im, G_re, G_im;
        fpr f_re, f_im, g_re, g_im;
        fpr a_re, a_im, b_re, b_im;

        F_re = F[u];
        F_im = F[u + hn];
        G_re = G[u];
        G_im = G[u + hn];
        f_re = f[u];
        f_im = f[u + hn];
        g_re = g[u];
        g_im = g[u + hn];

        FPC_MUL(a_re, a_im, F_re, F_im, f_re, fpr_neg(f_im));
        FPC_MUL(b_re, b_im, G_re, G_im, g_re, fpr_neg(g_im));
        d[u] = fpr_add(a_re, b_re);
        d[u + hn] = fpr_add(a_im, b_im);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_mul_autoadj_fft(
    fpr *a, const fpr *b, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        a[u] = fpr_mul(a[u], b[u]);
        a[u + hn] = fpr_mul(a[u + hn], b[u]);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_div_autoadj_fft(
    fpr *a, const fpr *b, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr ib;

        ib = fpr_inv(b[u]);
        a[u] = fpr_mul(a[u], ib);
        a[u + hn] = fpr_mul(a[u + hn], ib);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_LDL_fft(
    const fpr *g00,
    fpr *g01, fpr *g11, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr g00_re, g00_im, g01_re, g01_im, g11_re, g11_im;
        fpr mu_re, mu_im;

        g00_re = g00[u];
        g00_im = g00[u + hn];
        g01_re = g01[u];
        g01_im = g01[u + hn];
        g11_re = g11[u];
        g11_im = g11[u + hn];
        FPC_DIV(mu_re, mu_im, g01_re, g01_im, g00_re, g00_im);
        FPC_MUL(g01_re, g01_im, mu_re, mu_im, g01_re, fpr_neg(g01_im));
        FPC_SUB(g11[u], g11[u + hn], g11_re, g11_im, g01_re, g01_im);
        g01[u] = mu_re;
        g01[u + hn] = fpr_neg(mu_im);
    }
}

/* see inner.h */
void
PQCLEAN_FALCON512_NEON_poly_LDLmv_fft(
    fpr *d11, fpr *l10,
    const fpr *g00, const fpr *g01,
    const fpr *g11, unsigned logn) {
    size_t n, hn, u;

    n = (size_t)1 << logn;
    hn = n >> 1;
    for (u = 0; u < hn; u ++) {
        fpr g00_re, g00_im, g01_re, g01_im, g11_re, g11_im;
        fpr mu_re, mu_im;

        g00_re = g00[u];
        g00_im = g00[u + hn];
        g01_re = g01[u];
        g01_im = g01[u + hn];
        g11_re = g11[u];
        g11_im = g11[u + hn];
        FPC_DIV(mu_re, mu_im, g01_re, g01_im, g00_re, g00_im);
        FPC_MUL(g01_re, g01_im, mu_re, mu_im, g01_re, fpr_neg(g01_im));
        FPC_SUB(d11[u], d11[u + hn], g11_re, g11_im, g01_re, g01_im);
        l10[u] = mu_re;
        l10[u + hn] = fpr_neg(mu_im);
    }
}
