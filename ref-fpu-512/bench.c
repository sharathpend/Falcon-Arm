#include "inner.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include "util.h"
#include "config.h"

#if _APPLE_M1_ == 1
#include "m1cycles.h"
#endif

#define NTESTS 100000
#define FALCON_LOGN 10
#define FALCON_N (1 << FALCON_LOGN)
#define FALCON_Q 12289

// Result is nanosecond per call
#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);
#define CALC(start, stop, ntests) ((double)((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / ntests;

// Result is clock cycles
#define TIME_CYCLES(s) s = rdtsc();
#define CALC_CYCLES(start, stop, ntests) (stop - start) / ntests;

void test_FFT(fpr *f, unsigned logn)
{
    struct timespec start, stop;
    long long start_cc, stop_cc;
    long long ns_fft, ns_ifft;
    long long cc_fft, cc_ifft;
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 100;
    }
    /* =================================== */
    TIME(start);
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(FFT)(f, logn);
    }
    TIME_CYCLES(stop_cc);
    TIME(stop);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);
    ns_fft = CALC(start, stop, ntests);

    /* =================================== */
    TIME(start);
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(iFFT)(f, logn);
    }
    TIME(stop);
    TIME_CYCLES(stop_cc);

    /* =================================== */
    cc_ifft = CALC_CYCLES(start_cc, stop_cc, ntests);
    ns_ifft = CALC(start, stop, ntests);
    printf("FFT (us) %u: %lld - %lld | ", logn, ns_fft, ns_ifft);
    printf("%lld - %lld\n", cc_fft, cc_ifft);
    printf("=======\n");
}

void test_NTT(uint16_t *a, unsigned logn)
{
    struct timespec start, stop;
    long long start_cc, stop_cc;
    long long ns_fft, ns_ifft;
    long long cc_fft, cc_ifft;
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 100;
    }
    /* =================================== */
    TIME(start);
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        mq_NTT(a, logn);
    }
    TIME_CYCLES(stop_cc);
    TIME(stop);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);
    ns_fft = CALC(start, stop, ntests);

    /* =================================== */
    TIME(start);
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        mq_iNTT(a, logn);
    }
    TIME(stop);
    TIME_CYCLES(stop_cc);

    /* =================================== */
    cc_ifft = CALC_CYCLES(start_cc, stop_cc, ntests);
    ns_ifft = CALC(start, stop, ntests);
    printf("NTT (us) %u: %lld - %lld | ", logn, ns_fft, ns_ifft);
    printf("%lld - %lld\n", cc_fft, cc_ifft);
    printf("=======\n");
}

void test_poly_add(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 10)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_add)(c, a, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
}

void test_poly_sub(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 10)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_sub)(c, a, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
}

void test_poly_neg(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 10)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_neg)(c, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
}

void test_poly_adj_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 10)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_adj_fft)(c, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
}

void test_poly_mul_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 100;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_mul_fft)(c, a, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
}

void test_poly_invnorm2_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 7)
    {
        ntests = NTESTS * 100;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_invnorm2_fft)(c, a, b, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
}

void test_poly_mul_autoadj_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 10)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_mul_autoadj_fft)(c, a, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
}

void test_poly_LDL_fft(fpr *c, fpr *a, fpr *b, unsigned logn, char *string)
{
    long long start_cc, stop_cc;
    long long cc_fft;
    unsigned ntests = NTESTS;
    if (logn < 10)
    {
        ntests = NTESTS * 1000;
    }
    /* =================================== */
    TIME_CYCLES(start_cc);
    for (unsigned i = 0; i < ntests; i++)
    {
        Zf(poly_LDL_fft)(c, a, b, logn);
    }
    TIME_CYCLES(stop_cc);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, ntests);

    printf("%s, %u, %lld\n", string, logn, cc_fft);
    printf("=======\n");
}

int main()
{
    fpr f[FALCON_N], fa[FALCON_N], fb[FALCON_N], fc[FALCON_N];
    uint16_t a[FALCON_N];
    for (int i = 0; i < FALCON_N; i++)
    {
        double_t t;
        t = (double)i;
        f[i].v = t;
        fa[i].v = t;
        fb[i].v = t;
        fc[i].v = t;
        a[i] = rand() % FALCON_Q;
    }

#if _APPLE_M1_ == 1
    setup_rdtsc();
#endif

    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_FFT(f, i);
    }

    test_NTT(a, 9);
    test_NTT(a, 10);

    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_poly_add(fc, fa, fb, i, "poly_add");
        test_poly_sub(fc, fa, fb, i, "poly_sub");
        test_poly_neg(fc, fa, fb, i, "poly_neg");
        test_poly_adj_fft(fc, fa, fb, i, "poly_adj_fft");
        test_poly_mul_fft(fc, fa, fb, i, "poly_mul_fft");
        test_poly_invnorm2_fft(fc, fa, fb, i, "poly_invnorm2_fft");
        test_poly_mul_autoadj_fft(fc, fa, fb, i, "poly_mul_autoadj_fft");
        test_poly_LDL_fft(fc, fa, fb, i, "poly_LDL_fft");
    }

    return 0;
}

// Compile flags:
// gcc -o bench bench_fft.c fft.c fpr.c m1cycles.c vrfy.c common.c shake.c codec.c rng.c -O3; 
// sudo ./bench 