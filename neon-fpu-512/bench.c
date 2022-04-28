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

#if _APPLE_M1_ == 1
#include "m1cycles.h"
#endif

#define NTESTS 100000

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
        ZfN(FFT)(f, logn);
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
        ZfN(iFFT)(f, logn);
    }
    TIME(stop);
    TIME_CYCLES(stop_cc);

    /* =================================== */
    cc_ifft = CALC_CYCLES(start_cc, stop_cc, ntests);
    ns_ifft = CALC(start, stop, ntests);
    printf("FFT (us) %u: %lld - %lld\n", logn, ns_fft, ns_ifft);
    printf("FFT (cc) %u: %lld - %lld\n", logn, cc_fft, cc_ifft);
    printf("=======\n");
}

void test_NTT(int16_t *a)
{
    struct timespec start, stop;
    long long start_cc, stop_cc;
    long long ns_fft, ns_ifft;
    long long cc_fft, cc_ifft;

    const unsigned logn = FALCON_LOGN;

    TIME(start);
    TIME_CYCLES(start_cc);
    for (int i = 0; i < NTESTS; i++)
    {
        ZfN(poly_ntt)(a, 0);
    }
    TIME_CYCLES(stop_cc);
    TIME(stop);
    cc_fft = CALC_CYCLES(start_cc, stop_cc, NTESTS);
    ns_fft = CALC(start, stop, NTESTS);
    
    /* =================================== */
    TIME(start);
    TIME_CYCLES(start_cc);
    for (int i = 0; i < NTESTS; i++)
    {
        ZfN(poly_invntt)(a);
    }

    /* =================================== */
    TIME_CYCLES(stop_cc);
    TIME(stop);
    cc_ifft = CALC_CYCLES(start_cc, stop_cc, NTESTS);
    ns_ifft = CALC(start, stop, NTESTS);
    printf("NTT (us) %u: %lld - %lld\n", logn, ns_fft, ns_ifft);
    printf("NTT (cc) %u: %lld - %lld\n", logn, cc_fft, cc_ifft);
    printf("=======\n");
}

int main()
{
    fpr f[FALCON_N];
    int16_t a[FALCON_N];
    for (int i = 0; i < FALCON_N; i++)
    {
        f[i] = (double)i;
        a[i] = rand() % FALCON_Q;
    }

#if _APPLE_M1_ == 1
    setup_rdtsc();
#endif

    for (unsigned i = 0; i <= FALCON_LOGN; i++)
    {
        test_FFT(f, i);
    }

    test_NTT(a);

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
