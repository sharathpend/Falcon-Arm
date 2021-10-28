#include "vfpr.h"
#include "macro.h"
#include "fpr.h"
#include <stdio.h>
#include "params.h"
#include "util.h"
#include "sampler.h"

typedef int (*samplerZ)(void *ctx, fpr mu, fpr sigma);

#define START 25
#define DEBUG 0
#define FAKE_GAUSS 1

void prints(const char *string, fpr a, fpr b)
{
    printf("%s:\n%.20f\n%.20f\n", string, a, b);
}

void printv(const char *string, float64x2_t a)
{
    printf("%s:\n%.20f\n%.20f\n", string, a[0], a[1]);
}

int gauss(fpr a, fpr b)
{
    (void)a;
    (void)b;
    static int i = START;
    i = i * 1337 + 31;
    return i++;
}

int gauss_test(fpr a, fpr b)
{
    (void)a;
    (void)b;
    static int i = START;
    i = i * 1337 + 31;
    return i++;
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

    x0 = w2;
    x1 = w3;
    sigma = tree1[3];
#if DEBUG
    prints("01", w0, w1);
    prints("23", w2, w3);
#endif
#if FAKE_GAUSS
    w2 = gauss(x0, sigma);
    w3 = gauss(x1, sigma);
#else
    w2 = fpr_of(samp(samp_ctx, x0, sigma));
    w3 = fpr_of(samp(samp_ctx, x1, sigma));
#endif
    a_re = fpr_sub(x0, w2);
    a_im = fpr_sub(x1, w3);
    b_re = tree1[0];
    b_im = tree1[1];
    c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
    c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
    x0 = fpr_add(c_re, w0);
    x1 = fpr_add(c_im, w1);
    sigma = tree1[2];
#if FAKE_GAUSS
    w0 = gauss(x0, sigma);
    w1 = gauss(x1, sigma);
#else
    w0 = fpr_of(samp(samp_ctx, x0, sigma));
    w1 = fpr_of(samp(samp_ctx, x1, sigma));
#endif
#if DEBUG
    prints("02", w0, w2);
    prints("13", w1, w3);
#endif
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

#if DEBUG
    prints("z1-02", w0, w2);
    prints("z1-13", w1, w3);
#endif
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
#if DEBUG
    prints("02", w0, w2);
    prints("13", w1, w3);
#endif
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
#if FAKE_GAUSS
    w2 = y0 = gauss(x0, sigma);
    w3 = y1 = gauss(x1, sigma);
#else
    w2 = y0 = fpr_of(samp(samp_ctx, x0, sigma));
    w3 = y1 = fpr_of(samp(samp_ctx, x1, sigma));
#endif
    a_re = fpr_sub(x0, y0);
    a_im = fpr_sub(x1, y1);

    b_re = tree0[0];
    b_im = tree0[1];

    c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
    c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
    x0 = fpr_add(c_re, w0);
    x1 = fpr_add(c_im, w1);
    sigma = tree0[2];
#if FAKE_GAUSS
    w0 = gauss(x0, sigma);
    w1 = gauss(x1, sigma);
#else
    w0 = fpr_of(samp(samp_ctx, x0, sigma));
    w1 = fpr_of(samp(samp_ctx, x1, sigma));
#endif
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

void sampling_neon(samplerZ samp, void *samp_ctx, fpr *z0, fpr *z1, const fpr *tree, const fpr *t0, const fpr *t1)
{
    const fpr *tree0, *tree1;
    tree0 = tree + 4;
    tree1 = tree + 8;

    // ------------------

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
    s_w2 = fpr_of(samp(samp_ctx, s_x0, sigma));
    s_w3 = fpr_of(samp(samp_ctx, s_x1, sigma));
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
    s_w0 = fpr_of(samp(samp_ctx, s_x0, sigma));
    s_w1 = fpr_of(samp(samp_ctx, s_x1, sigma));
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
    s_w2 = fpr_of(samp(samp_ctx, s_x0, sigma));
    s_w3 = fpr_of(samp(samp_ctx, s_x1, sigma));
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
    s_w0 = fpr_of(samp(samp_ctx, s_x0, sigma));
    s_w1 = fpr_of(samp(samp_ctx, s_x1, sigma));
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
}

int mem_compare(fpr a[4], fpr b[4], const char *string)
{
    printf("%s:\n", string);
    int ret = 0;
    for (int i = 0; i < 4; i++)
    {
        if (a[i] != b[i])
        {
            printf("%d: %.20f != %.20f\n", i, a[i], b[i]);
            ret = 1;
        }
    }
    if (!ret)
        printf("OK\n");
    return ret;
}

int main()
{
    fpr t0[4], t1[4], tree[8];
    fpr z0[4], z1[4], z2[4], z3[4];
    srand(0);
    for (int i = 0; i < 4; i++)
    {
        t0[i] = fRand(0, 5);
        t1[i] = fRand(0, 5);
        tree[i] = fRand(0, 5);
        tree[4 + i] = fRand(0, 5);
        // t0[i] = i;
        // t1[i] = i;
        // tree[i] = i;
        // tree[4 + i] = i;
    }

    sampler_context spc, spc_test;
    samplerZ samp;
    char buf[20], buf_test[20];
    void *samp_ctx, *samp_ctx_test;

    inner_shake256_context rng, rng_test;
    const int logn = 9;
    int ret = 0;

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

    inner_shake256_init(&rng_test);
    inner_shake256_inject(&rng_test, (uint8_t *)buf_test, strlen(buf_test));
    inner_shake256_flip(&rng_test);

    PQCLEAN_FALCON512_NEON_prng_init(&spc.p, &rng);
    PQCLEAN_FALCON512_NEON_prng_init(&spc_test.p, &rng_test);
    samp = PQCLEAN_FALCON512_NEON_sampler;
    samp_ctx = &spc;
    samp_ctx_test = &spc_test;

    printf("=========1\n");
    sampling_original(samp, samp_ctx, z0, z1, tree, t0, t1);
    printf("=========2\n");
    sampling_neon(samp, samp_ctx_test, z2, z3, tree, t0, t1);
    printf("===========\n");

    ret |= mem_compare(z0, z2, "z0");
    ret |= mem_compare(z1, z3, "z1");

    return ret;
}
