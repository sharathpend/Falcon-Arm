#include "inner.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "util.h"
#include "config.h"
#include "api.h"
#include "poly.h"

#define ITERATIONS 100000
uint64_t times[ITERATIONS];

#if BENCH_CYCLES == 1

#if APPLE_M1 == 1

// Result is cycle per call
#include "m1cycles.h"

#define TIME(s) s = rdtsc();
#define CALC(start, stop, ntests) (stop - start) / ntests;
#else

// Result is cycle per call
#include "hal.h"

#define TIME(s) s = hal_get_time();
#define CALC(start, stop, ntests) (stop - start) / ntests;
#endif

#else

#include <time.h>
// Result is nanosecond per call

#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);
#define CALC(start, stop, ntests) ((double)((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / ntests;
#endif

void print_header()
{
    printf("\n| Function | logn | cycles |\n");
    printf("|:-------------|----------:|-----------:|\n");
}

static int cmp_uint64_t(const void *a, const void *b)
{
    return (int)((*((const uint64_t *)a)) - (*((const uint64_t *)b)));
}

void test_FFT(fpr *f, unsigned logn)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft, ifft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(FFT)(f, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(iFFT)(f, logn);
        TIME(stop);

        times[i] = stop - start;
    }

    /* =================================== */
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);    
    ifft = times[ITERATIONS >> 1];
    printf("| FFT %u | %8lld | %8lld\n", logn, fft, ifft);
}

void test_NTT(int16_t *a, unsigned logn)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft, ifft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_ntt)(a, NTT_NONE);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_invntt)(a, INVNTT_NONE);
        TIME(stop);

        times[i] = stop - start;
    }

    /* =================================== */
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    ifft = times[ITERATIONS >> 1];
    printf("| NTT %u | %8lld | %8lld\n", logn, fft, ifft);
}

void test_poly_add(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_add)(c, a, b, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_sub(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_sub)(c, a, b, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_neg(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_neg)(c, a, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_adj_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_adj_fft)(c, a, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_mul_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_mul_fft)(c, a, b, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_invnorm2_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_invnorm2_fft)(c, a, b, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_mul_autoadj_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_mul_autoadj_fft)(c, a, b, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_LDL_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_LDL_fft)(c, a, b, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_LDLmv_fft(fpr *d11, fpr *l01, const fpr *c, const fpr *a, const fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_LDLmv_fft)(d11, l01, c, a, b, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_split_fft(fpr *restrict f0, fpr *restrict f1,
                         const fpr *restrict f, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_split_fft)(f0, f1, f, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_poly_merge_fft(fpr *restrict f, const fpr *restrict f0,
                         const fpr *restrict f1, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(poly_merge_fft)(f, f0, f1, logn);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
}

void test_compute_bnorm(fpr *restrict fa, fpr *restrict fb, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = ITERATIONS;
    
    /* =================================== */
    for (unsigned i = 0; i < ntests; i++)
    {
        TIME(start);
        ZfN(compute_bnorm)(fa, fb);
        TIME(stop);

        times[i] = stop - start;
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    fft = times[ITERATIONS >> 1];

    printf("| %8s | %8u | %8lld\n", string, FALCON_LOGN, fft);
}

// Test Gaussian0 Sampler
// void test_gaussian0_sampler()
// {
// #if BENCH_CYCLES == 0
//     struct timespec start, stop;
// #else
//     long long start, stop;
// #endif
//     long long result;
//     unsigned ntests = ITERATIONS;
    
//     // Initialize RNG
//     inner_shake256_context rng;
//     prng p;
//     inner_shake256_init(&rng);
//     Zf(prng_init)(&p, &rng);
    
//     /* =================================== */
//     for (unsigned i = 0; i < ntests; i++)
//     {
//         TIME(start);
//         ZfN(gaussian0_sampler)(&p);
//         TIME(stop);

//         times[i] = stop - start;
//     }
//     qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
//     result = times[ITERATIONS >> 1];

//     printf("| %8s | %8u | %8lld\n", "gauss0", FALCON_LOGN, result);
// }


// Test Full Sampler 1 run
void test_sampler_single_run()
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long result;
    unsigned ntests = ITERATIONS;
    
    // Initialize sampler context
    sampler_context spc;
    inner_shake256_context rng;
    inner_shake256_init(&rng);
    
    // Use specific seed: 0x1570F5400B5D4105A9AD59
    //uint8_t seed_data[] = {0x15, 0x70, 0xF5, 0x40, 0x0B, 0x5D, 0x41, 0x05, 0xA9, 0xAD, 0x59};
    //inner_shake256_inject(&rng, seed_data, sizeof(seed_data));
    //inner_shake256_flip(&rng);
    
    Zf(prng_init)(&spc.p, &rng);
    
#if FALCON_LOGN == 9
    spc.sigma_min = fpr_sigma_min_9;
#else
    spc.sigma_min = fpr_sigma_min_10;
#endif

    // mu_min = -75.0
    // mu_max = 75.0
    // sigma_min = 1.277833697
    // sigma_max = 1.8205

    double mu_min = -75.0, mu_max = 75.0;
    double sigma_min = 1.277833697, sigma_max = 1.8205;

    
    
    // fpr mu = fpr_of(-44.301977378143064);
    // fpr isigma = fpr_of(1.767660377221966);

    
    /* =================================== */
    for (unsigned i = 0; i <1; i++)
    {
        // Uniform random double in [0,1)
        double urand = (double)rand() / ((double)RAND_MAX + 1.0);

        // Uniform mu in [mu_min, mu_max]
        double mu_val = mu_min + urand * (mu_max - mu_min);

        // Uniform sigma in [sigma_min, sigma_max]
        urand = (double)rand() / ((double)RAND_MAX + 1.0);
        double sigma_val = sigma_min + urand * (sigma_max - sigma_min);

        fpr mu = fpr_of(mu_val);
        fpr sigma = fpr_of(sigma_val);
        fpr isigma = 1/sigma;

        TIME(start);
        Zf(sampler)(&spc, mu, isigma);
        TIME(stop);

// #if BENCH_CYCLES == 0
//         times[i] = (uint64_t)((stop.tv_sec - start.tv_sec) * 1000000000LL + (stop.tv_nsec - start.tv_nsec));
// #else
        times[i] = stop - start;
// #endif
    }
    //qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    //result = times[ITERATIONS >> 1];
    result = times[0];
    printf("| %8s | %8u | %8lld \n", "sampler", FALCON_LOGN, result);
}



// Test Full Sampler
void test_sampler()
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long result;
    unsigned ntests = ITERATIONS;
    
    // Initialize sampler context
    sampler_context spc;
    inner_shake256_context rng;
    inner_shake256_init(&rng);
    
    // Use specific seed: 0x1570F5400B5D4105A9AD59
    //uint8_t seed_data[] = {0x15, 0x70, 0xF5, 0x40, 0x0B, 0x5D, 0x41, 0x05, 0xA9, 0xAD, 0x59};
    //inner_shake256_inject(&rng, seed_data, sizeof(seed_data));
    //inner_shake256_flip(&rng);
    
    Zf(prng_init)(&spc.p, &rng);
    
#if FALCON_LOGN == 9
    spc.sigma_min = fpr_sigma_min_9;
#else
    spc.sigma_min = fpr_sigma_min_10;
#endif

    // mu_min = -75.0
    // mu_max = 75.0
    // sigma_min = 1.277833697
    // sigma_max = 1.8205

    double mu_min = -75.0, mu_max = 75.0;
    double sigma_min = 1.277833697, sigma_max = 1.8205;

    
    
    // fpr mu = fpr_of(-44.301977378143064);
    // fpr isigma = fpr_of(1.767660377221966);

    
    /* =================================== */
    for (unsigned i = 0; i <ITERATIONS; i++)
    {
        // Uniform random double in [0,1)
        double urand = (double)rand() / ((double)RAND_MAX + 1.0);

        // Uniform mu in [mu_min, mu_max]
        double mu_val = mu_min + urand * (mu_max - mu_min);

        // Uniform sigma in [sigma_min, sigma_max]
        urand = (double)rand() / ((double)RAND_MAX + 1.0);
        double sigma_val = sigma_min + urand * (sigma_max - sigma_min);

        fpr mu = fpr_of(mu_val);
        fpr sigma = fpr_of(sigma_val);
        fpr isigma = 1/sigma;

        TIME(start);
        Zf(sampler)(&spc, mu, isigma);
        TIME(stop);

// #if BENCH_CYCLES == 0
//         times[i] = (uint64_t)((stop.tv_sec - start.tv_sec) * 1000000000LL + (stop.tv_nsec - start.tv_nsec));
// #else
        times[i] = stop - start;
// #endif
    }
    qsort(times, ITERATIONS, sizeof(uint64_t), cmp_uint64_t);
    result = times[ITERATIONS >> 1];
    // result = times[0];
    printf("| %8s | %8u | %8lld \n", "sampler", FALCON_LOGN, result);
}

int main()
{
    fpr f[FALCON_N], fa[FALCON_N], fb[FALCON_N], fc[FALCON_N], tmp[FALCON_N] = {0};
    int16_t a[FALCON_N];
    for (int i = 0; i < FALCON_N; i++)
    {
        double_t t;
        t = (double)i;
        f[i] = t;
        fa[i] = t;
        fb[i] = t;
        fc[i] = t;
        a[i] = rand() % FALCON_Q;
    }

#if BENCH_CYCLES == 1
#if APPLE_M1 == 1
    setup_rdtsc();
#endif
#endif

    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_FFT(f, i);
    // }

    // test_NTT(a, FALCON_LOGN);

    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_add(fc, fa, fb, i, "poly_add");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_sub(fc, fa, fb, i, "poly_sub");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_neg(fc, fa, fb, i, "poly_neg");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_adj_fft(fc, fa, fb, i, "poly_adj_fft");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_mul_fft(fc, fa, fb, i, "poly_mul_fft");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_invnorm2_fft(fc, fa, fb, i, "poly_invnorm2_fft");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_mul_autoadj_fft(fc, fa, fb, i, "poly_mul_autoadj_fft");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_LDL_fft(fc, fa, fb, i, "poly_LDL_fft");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_LDLmv_fft(f, tmp, fc, fa, fb, i, "poly_LDLmv_fft");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_split_fft(fa, fb, f, i, "poly_split_fft");
    // }
    // print_header();
    // for (unsigned i = 0; i <= FALCON_LOGN; i++)
    // {
    //     test_poly_merge_fft(f, fa, fb, i, "poly_merge_fft");
    // }
    // print_header();
    // test_compute_bnorm(fa, fb, "compute_bnorm");
    
    print_header();
    // test_gaussian0_sampler();
    // test_sampler();

    test_sampler_single_run();

    // print_header();
    // test_mul();

    return 0;
}
