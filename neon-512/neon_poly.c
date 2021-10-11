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
