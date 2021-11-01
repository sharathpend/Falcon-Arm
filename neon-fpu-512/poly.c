#include "inner.h"
#include "macrofx4.h"

/* see inner.h */
void Zf(poly_add)(fpr *c, const fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    float64x2x4_t neon_a, neon_b, neon_c;
    const int falcon_n = 1 << logn;
    for (int i = 0; i < falcon_n; i += 8)
    {
        vloadx4(neon_a, &a[i]);
        vloadx4(neon_b, &b[i]);

        vfaddx4(neon_c, neon_a, neon_b);

        vstorex4(&c[i], neon_c);
    }
}

/* see inner.h */
/* 
 * c = a - b
 */
void Zf(poly_sub)(fpr *c, const fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    float64x2x4_t neon_a, neon_b, neon_c;
    const int falcon_n = 1 << logn;
    for (int i = 0; i < falcon_n; i += 8)
    {
        vloadx4(neon_a, &a[i]);
        vloadx4(neon_b, &b[i]);

        vfsubx4(neon_c, neon_a, neon_b);

        vstorex4(&c[i], neon_c);
    }
}

/* see inner.h */
/* 
 * c = -a 
 */
void Zf(poly_neg)(fpr *c, const fpr *restrict a, unsigned logn)
{
    float64x2x4_t neon_a, neon_c;
    const int falcon_n = 1 << logn;
    for (int i = 0; i < falcon_n; i += 8)
    {
        vloadx4(neon_a, &a[i]);

        vfnegx4(neon_c, neon_a);

        vstorex4(&c[i], neon_c);
    }
}

/* see inner.h */
void Zf(poly_adj_fft)(fpr *c, const fpr *restrict a, unsigned logn)
{

    float64x2x4_t neon_a, neon_c;
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    for (int i = hn; i < falcon_n; i += 8)
    {
        vloadx4(neon_a, &a[i]);

        vfnegx4(neon_c, neon_a);

        vstorex4(&c[i], neon_c);
    }
}

/* see inner.h */
/* 
 * c = a * b
 */
void Zf(poly_mul_fft)(fpr *restrict c, const fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    // Total 32 registers
    float64x2x4_t a_re, b_re, a_im, b_im; // 24
    float64x2x4_t c_re, c_im;             // 8
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(b_re, &b[i]);
        vloadx4(a_im, &a[i + hn]);
        vloadx4(b_im, &b[i + hn]);

        vfmulx4(c_re, a_re, b_re);
        vfmsx4(c_re, c_re, a_im, b_im);

        vfmulx4(c_im, a_re, b_im);
        vfmax4(c_im, c_im, a_im, b_re);

        vstorex4(&c[i], c_re);
        vstorex4(&c[i + hn], c_im);
    }
}

/* see inner.h */
void Zf(poly_muladj_fft)(fpr *d, fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    float64x2x4_t a_re, b_re, d_re, a_im, b_im, d_im; // 24
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    for (int i = 0; i < falcon_n; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(b_re, &b[i]);
        vloadx4(a_im, &a[i + hn]);
        vloadx4(b_im, &b[i + hn]);

        vfmulx4(d_re, a_re, b_re);
        vfmax4(d_re, d_re, a_im, b_im);

        vfmulx4(d_im, a_im, b_re);
        vfmsx4(d_im, d_im, a_re, b_im);

        vstorex4(&d[i], d_re);
        vstorex4(&d[i + hn], d_im);
    }
}

/* see inner.h */
void Zf(poly_mulselfadj_fft)(fpr *c, const fpr *restrict a, unsigned logn)
{
    /*
	 * Since each coefficient is multiplied with its own conjugate,
	 * the result contains only real values.
	 */
    float64x2x4_t a_re, a_im, c_re, c_im; // 16
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;

    vfdupx4(c_im, 0);

    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(a_im, &a[i + hn]);

        vfmulx4(c_re, a_re, a_re);
        vfmax4(c_re, c_re, a_im, a_im);

        vstorex4(&c[i], c_re);
        vstorex4(&c[i + hn], c_im);
    }
}

/* see inner.h */
/* 
 * c = a * scalar_x
 */
void Zf(poly_mulconst)(fpr *c, const fpr *a, const fpr x, unsigned logn)
{
    // Total 9 registers
    float64x2x4_t neon_a, neon_c;
    const int falcon_n = 1 << logn;
    for (int i = 0; i < falcon_n; i += 8)
    {
        vloadx4(neon_a, &a[i]);

        vfmulnx4(neon_c, neon_a, x);

        vstorex4(&c[i], neon_c);
    }
}

/* see inner.h */
void Zf(poly_div_fft)(fpr *restrict c, const fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    float64x2x4_t a_re, a_im, b_re, b_im, c_re, c_im, m;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(b_re, &b[i]);
        vloadx4(a_im, &a[i + hn]);
        vloadx4(b_im, &b[i + hn]);

        vfmulx4(m, b_re, b_re);
        vfmax4(m, m, b_im, b_im);
        vfinvx4(m, m);

        vfmulx4(c_re, a_re, b_re);
        vfmax4(c_re, c_re, a_im, b_im);

        vfmulx4(c_im, a_im, b_re);
        vfmsx4(c_im, c_im, a_re, b_im);

        vfmulx4(c_re, c_re, m);
        vfmulx4(c_im, c_im, m);

        vstorex4(&c[i], c_re);
        vstorex4(&c[i + hn], c_im);
    }
}

/* see inner.h */
void Zf(poly_invnorm2_fft)(fpr *restrict d, const fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    float64x2x4_t a_re, a_im, b_re, b_im, c_re, c_im, d_re;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(a_im, &a[i + hn]);
        vloadx4(b_re, &b[i]);
        vloadx4(b_im, &b[i + hn]);

        vfmulx4(c_re, a_re, a_re);
        vfmax4(c_re, c_re, a_im, a_im);
        
        vfmulx4(c_im, b_re, b_re);
        vfmax4(c_im, c_im, b_im, b_im);

        vfaddx4(d_re, c_re, c_im);
        vfinvx4(d_re, d_re);

        vstorex4(&d[i], d_re);
    }
}

/* see inner.h */
void Zf(poly_add_muladj_fft)(fpr *restrict d,
                             const fpr *restrict F, const fpr *restrict G,
                             const fpr *restrict f, const fpr *restrict g, unsigned logn)
{
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    float64x2x4_t F_re, F_im, G_re, G_im;
    float64x2x4_t f_re, f_im, g_re, g_im;
    float64x2x4_t a_re, a_im, b_re, b_im;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(F_re, &F[i]);
        vloadx4(F_im, &F[i + hn]);
        vloadx4(G_re, &G[i]);
        vloadx4(G_im, &G[i + hn]);
        vloadx4(f_re, &f[i]);
        vloadx4(f_im, &f[i + hn]);
        vloadx4(g_re, &g[i]);
        vloadx4(g_im, &g[i + hn]);

        vfmulx4(a_re, F_re, f_re);
        vfmax4(a_re, a_re, F_im, f_im);

        vfmulx4(a_im, F_im, f_re);
        vfmsx4(a_im, a_im, F_re, f_im);

        vfmulx4(b_re, G_re, g_re);
        vfmax4(b_re, b_re, G_im, g_im);

        vfmulx4(b_im, G_im, g_re);
        vfmsx4(b_im, b_im, G_re, g_im);

        vfsubx4(a_re, a_re, b_re);
        vfsubx4(a_im, a_im, b_im);

        vstorex4(&d[i], a_re);
        vstorex4(&d[i + hn], a_im);
    }
}

/* see inner.h */
void Zf(poly_mul_autoadj_fft)(fpr *restrict c, const fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    float64x2x4_t a_re, a_im, b_re, c_re, c_im;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(a_im, &a[i + hn]);
        vloadx4(b_re, &b[i]);

        vfmulx4(c_re, a_re, b_re);
        vfmulx4(c_im, a_im, b_re);

        vstorex4(&c[i], c_re);
        vstorex4(&c[i + hn], c_im);
    }
}

/* see inner.h */
void Zf(poly_div_autoadj_fft)(fpr *restrict c, const fpr *restrict a, const fpr *restrict b, unsigned logn)
{
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    float64x2x4_t a_re, a_im, b_re, binv, c_re, c_im;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(a_re, &a[i]);
        vloadx4(a_im, &a[i + hn]);
        vloadx4(b_re, &b[i]);

        vfinvx4(binv, b_re);

        vfmulx4(c_re, a_re, binv);
        vfmulx4(c_im, a_im, binv);

        vstorex4(&c[i], c_re);
        vstorex4(&c[i + hn], c_im);
    }
}

/* see inner.h */
void Zf(poly_LDL_fft)(const fpr *restrict g00, fpr *restrict g01, fpr *restrict g11, unsigned logn)
{
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    float64x2x4_t g00_re, g00_im, g01_re, g01_im, g11_re, g11_im;
    float64x2x4_t mu_re, mu_im, m, d_re, d_im;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(g00_re, &g00[i]);
        vloadx4(g00_im, &g00[i + hn]);
        vloadx4(g01_re, &g01[i]);
        vloadx4(g01_im, &g01[i + hn]);
        vloadx4(g11_re, &g11[i]);
        vloadx4(g11_im, &g11[i + hn]);

        vfmulx4(m, g00_re, g00_re);
        vfmax4(m, m, g00_im, g00_im);
        vfinvx4(m, m);

        vfmulx4(mu_re, g01_re, g00_re);
        vfmax4(mu_re, mu_re, g01_im, g00_im);

        vfmulx4(mu_im, g01_im, g00_re);
        vfmsx4(mu_im, mu_im, g01_re, g00_im);

        vfmulx4(mu_re, mu_re, m);
        vfmulx4(mu_im, mu_im, m);

        vfmulx4(d_re, mu_re, g01_re);
        vfmax4(d_re, d_re, mu_im, g01_im);

        vfmulx4(d_im, mu_im, g01_re);
        vfmsx4(d_im, d_im, mu_re, g01_im);

        vfsubx4(g11_re, g11_re, d_re);
        vfsubx4(g11_im, g11_im, d_re);

        vstorex4(&g11[i], g11_re);
        vstorex4(&g11[i + hn], g11_im);

        vfnegx4(mu_im, mu_im);

        vstorex4(&g01[i], mu_re);
        vstorex4(&g01[i + hn], mu_im);
    }
}

/* see inner.h */
void Zf(poly_LDLmv_fft)(fpr *restrict d11, fpr *restrict l10,
                        const fpr *restrict g00, const fpr *restrict g01,
                        const fpr *restrict g11, unsigned logn)
{
    const int falcon_n = 1 << logn;
    const int hn = falcon_n >> 1;
    float64x2x4_t g00_re, g00_im, g01_re, g01_im,
        g11_re, g11_im, mu_re, mu_im, m;
    for (int i = 0; i < hn; i += 8)
    {
        vloadx4(g00_re, &g00[i]);
        vloadx4(g00_im, &g00[i + hn]);
        vloadx4(g01_re, &g01[i]);
        vloadx4(g01_im, &g01[i + hn]);
        vloadx4(g11_re, &g11[i]);
        vloadx4(g11_im, &g11[i + hn]);

        vfmulx4(m, g00_re, g00_re);
        vfmax4(m, m, g00_im, g00_im);
        vfinvx4(m, m);

        vfmulx4(mu_re, g01_re, g00_re);
        vfmax4(mu_re, mu_re, g01_im, g00_im);

        vfmulx4(mu_im, g01_im, g00_re);
        vfmsx4(mu_im, mu_im, g01_re, g00_im);

        vfmulx4(mu_re, mu_re, m);
        vfmulx4(mu_im, mu_im, m);

        vfmulx4(g01_re, mu_re, g01_re);
        vfmax4(g01_re, g01_re, mu_im, g01_im);

        vfmulx4(g01_im, mu_im, g01_re);
        vfmsx4(g01_im, g01_im, mu_re, g01_im);

        vfsubx4(g11_re, g11_re, g01_re);
        vfsubx4(g11_im, g11_im, g01_im);

        vstorex4(&d11[i], g11_re);
        vstorex4(&d11[i + hn], g11_im);

        vfnegx4(mu_im, mu_im);

        vstorex4(&l10[i], mu_re);
        vstorex4(&l10[i + hn], mu_im);
    }
}
