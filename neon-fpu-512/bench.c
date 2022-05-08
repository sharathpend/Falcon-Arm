#include "inner.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include "util.h"
#include "config.h"
#include "api.h"
#include "poly.h"

#if BENCH_CYCLES == 1
#if APPLE_M1 == 1
#include "m1cycles.h"

// Result is cycle per call
#define TIME(s) s = rdtsc();
#define CALC(start, stop, ntests) (stop - start) / ntests;
#else 
#include "hal.h"

// Result is cycle per call
#define TIME(s) s = hal_get_time();
#define CALC(start, stop, ntests) (stop - start) / ntests;
#endif 

#else
// Result is nanosecond per call
#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);
#define CALC(start, stop, ntests) ((double)((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / ntests;
#endif

#define NTESTS 10000

uint64_t get_cycle_count()
{
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

void print_header()
{
    printf("\n| Function | logn | cycles |\n");
    printf("|:-------------|----------:|-----------:|\n");
}

void test_FFT(fpr *f, unsigned logn)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft, ifft;
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(FFT)(f, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(iFFT)(f, logn);
    }
    TIME(stop);

    /* =================================== */
    ifft = CALC(start, stop, ntests);
    printf("FFT %u: %lld - %lld\n", logn, fft, ifft);
    printf("=======\n");
}

void test_NTT(int16_t *a, unsigned logn)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft, ifft;
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_ntt)(a, 0);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_invntt)(a);
    }
    TIME(stop);

    /* =================================== */
    ifft = CALC(start, stop, ntests);
    printf("NTT %u: %lld - %lld\n", logn, fft, ifft);
    printf("=======\n");
}

void test_poly_add(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
#if BENCH_CYCLES == 0
    struct timespec start, stop;
#else
    long long start, stop;
#endif
    long long fft;
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_add)(c, a, b, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_sub)(c, a, b, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_neg)(c, a, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_adj_fft)(c, a, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_mul_fft)(c, a, b, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_invnorm2_fft)(c, a, b, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_mul_autoadj_fft)(c, a, b, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_LDL_fft)(c, a, b, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_LDLmv_fft)(d11, l01, c, a, b, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_split_fft)(f0, f1, f, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

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
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME(start);
    for (unsigned i = 0; i < ntests; i++)
    {
        ZfN(poly_merge_fft)(f, f0, f1, logn);
    }
    TIME(stop);
    fft = CALC(start, stop, ntests);

    printf("| %8s | %8u | %8lld\n", string, logn, fft);
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

    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_FFT(f, i);
    }

    test_NTT(a, FALCON_LOGN);
    
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_add(fc, fa, fb, i, "poly_add");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_sub(fc, fa, fb, i, "poly_sub");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_neg(fc, fa, fb, i, "poly_neg");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_adj_fft(fc, fa, fb, i, "poly_adj_fft");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_mul_fft(fc, fa, fb, i, "poly_mul_fft");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_invnorm2_fft(fc, fa, fb, i, "poly_invnorm2_fft");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_mul_autoadj_fft(fc, fa, fb, i, "poly_mul_autoadj_fft");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_LDL_fft(fc, fa, fb, i, "poly_LDL_fft");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_LDLmv_fft(f, tmp, fc, fa, fb, i, "poly_LDLmv_fft");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_split_fft(fa, fb, f, i, "poly_split_fft");
    }
    print_header();
    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_merge_fft(f, fa, fb, i, "poly_merge_fft");
    }

    return 0;
}

/*
 * Result in nanosection
 * FFT          - iFFT
 *  9: 4609     - 4333
 * 10: 11389    - 10790
 *
 * NTT          - iNTT
 *  9: 2560     - 2646
 * 10: 5426     - 5721
 */
