#include "vfpr.h"
#include "macro.h"
#include "fpr.h"
#include <stdio.h>
#include "params.h"
#include "util.h"
#include "sampler.h"

typedef int (*samplerZ)(void *ctx, fpr mu, fpr sigma);

void print(const char *string, fpr a, fpr b)
{
    printf("%s:\n%.20f\n%.20f\n", string, a, b);
}

void printv(const char *string, float64x2_t a)
{
    printf("%s:\n%.20f\n%.20f\n", string, a[0], a[1]);
}

void sampling_original(samplerZ samp, void *samp_ctx, fpr *z0, fpr *z1, const fpr *tree, const fpr *t0, const fpr *t1)
{
    const fpr *tree0, *tree1;
    // ----------------------
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

    // print("w01", w0, w1);
    // print("w23", w2, w3);
}

void sampling_neon(samplerZ samp, void *samp_ctx, fpr *z0, fpr *z1, const fpr *tree, const fpr *t0, const fpr *t1)
{
    const fpr *tree0, *tree1;
    tree0 = tree + 4;
    tree1 = tree + 8;

    // ------------------

    float64x2x2_t tmp;
    float64x2_t a, b, c, w01, w23, x01, x23;
    float64x1_t c_re, c_im;

    vload2(tmp, &t1[0]);
    a = tmp.val[0]; // a_re, a_im
    b = tmp.val[1]; // b_re, b_im

    // z = a_re + b_re | a_im + b_im
    c = vfpr_add(a, b);
    w01 = vfpr_half(c);

    // z = a_re - b_re | a_im - b_im
    c = vfpr_sub(a, b);
    c_re = vget_low_f64(c);
    c_im = vget_high_f64(c);

    c = vcombine_f64(vadd_f64(c_re, c_im), vsub_f64(c_im, c_re));
    w23 = vfpr_mul(c, vdupq_n_f64(fpr_invsqrt8));

    // printv("w01", w01);
    // printv("w23", w23);

    double s_x0, s_x1, sigma, s_w2, s_w3;

    x01 = w23;
    s_x0 = vgetq_lane_f64(w23, 0);
    s_x1 = vgetq_lane_f64(w23, 1);
    sigma = tree1[3];
    s_w2 = fpr_of(samp(samp_ctx, s_x0, sigma));
    s_w3 = fpr_of(samp(samp_ctx, s_x1, sigma));
    w23 = vsetq_lane_f64(s_w2, w23, 0);
    w23 = vsetq_lane_f64(s_w3, w23, 1);
    a = vfpr_sub(x01, w23);
    b = vld1q_f64(&tree1[0]);
    // a_re * b_re | a_im * b_im 
    c = vmulq_f64(a, b);

}

int main()
{
    fpr t0[4], t1[4], tree[8];
    fpr z0[4], z1[4];
    for (int i = 0; i < 4; i++)
    {
        t0[i] = fRand(-FALCON_N, FALCON_N);
        t1[i] = fRand(-FALCON_N, FALCON_N);
        tree[i] = fRand(-FALCON_N, FALCON_N);
        tree[4 + i] = fRand(-FALCON_N, FALCON_N);
    }

    sampler_context spc;
    samplerZ samp;
    char buf[20];
    void *samp_ctx;

    inner_shake256_context rng;
    const int logn = 9;

    if (logn == 10)
    {
        spc.sigma_min = fpr_sigma_min_10;
    }
    else
    {
        spc.sigma_min = fpr_sigma_min_9;
    }
    inner_shake256_init(&rng);
	inner_shake256_inject(&rng, (uint8_t *)buf, strlen(buf));
	inner_shake256_flip(&rng);

    printf("=========1\n");
    PQCLEAN_FALCON512_NEON_prng_init(&spc.p, &rng);
    printf("=========2\n");
    samp = PQCLEAN_FALCON512_NEON_sampler;
    samp_ctx = &spc;

    sampling_original(samp, samp_ctx, z0, z1, tree, t0, t1);
    printf("===========\n");
    sampling_neon(samp, samp_ctx, z0, z1, tree, t0, t1);

    return 0;
}
