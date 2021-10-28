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
    // ----------------------
    fpr x0, x1, y0, y1, sigma;
    fpr a_re, a_im, b_re, b_im, c_re, c_im;

    x0 = t1[0];
    x1 = t1[1];
    sigma = tree[3];
#if FAKE_GAUSS
    z1[0] = y0 = fpr_of(gauss(x0, sigma));
    z1[1] = y1 = fpr_of(gauss(x1, sigma));
#else
    z1[0] = y0 = fpr_of(samp(samp_ctx, x0, sigma));
    z1[1] = y1 = fpr_of(samp(samp_ctx, x1, sigma));
#endif
    a_re = fpr_sub(x0, y0);
    a_im = fpr_sub(x1, y1);
    b_re = tree[0];
    b_im = tree[1];
    c_re = fpr_sub(fpr_mul(a_re, b_re), fpr_mul(a_im, b_im));
    c_im = fpr_add(fpr_mul(a_re, b_im), fpr_mul(a_im, b_re));
    x0 = fpr_add(c_re, t0[0]);
    x1 = fpr_add(c_im, t0[1]);
    sigma = tree[2];
#if FAKE_GAUSS
    z0[0] = fpr_of(gauss(x0, sigma));
    z0[1] = fpr_of(gauss(x1, sigma));
#else
    z0[0] = fpr_of(samp(samp_ctx, x0, sigma));
    z0[1] = fpr_of(samp(samp_ctx, x1, sigma));
#endif

    return;
}

void sampling_neon(samplerZ samp, void *samp_ctx, fpr *z0, fpr *z1, const fpr *tree, const fpr *t0, const fpr *t1)
{
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
