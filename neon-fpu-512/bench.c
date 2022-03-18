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

#define NTESTS 1000000
#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);;
// Result is nanosecond per call 
#define  CALC(start, stop) ((double) ((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / NTESTS;



void test_FFT()
{
    struct timespec start, stop;
    long long ns_fft, ns_ifft;

    const unsigned logn = FALCON_LOGN;

    fpr f[FALCON_N];
    for (int i = 0; i < FALCON_N; i++)
    {
        f[i] = (double) i;
    }
    /* =================================== */
    
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        ZfN(FFT)(f, logn);
    }
    TIME(stop);
    ns_fft = CALC(start, stop);
    /* =================================== */
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        ZfN(iFFT)(f, logn);
    }
    /* =================================== */
    TIME(stop);
    ns_ifft = CALC(start, stop);
    printf("FFT %u: %8lld - %8lld\n", logn, ns_fft, ns_ifft);
}

void test_NTT()
{
    struct timespec start, stop;
    long long ns_fft, ns_ifft;

    const unsigned logn = FALCON_LOGN;

    int16_t a[FALCON_N];
    for (int i = 0; i < FALCON_N; i++)
    {
        a[i] = rand() % FALCON_Q;
    }

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        ZfN(poly_ntt)(a, 0);
    }
    TIME(stop);
    ns_fft = CALC(start, stop);
    /* =================================== */
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        ZfN(poly_invntt)(a);
    }
    /* =================================== */
    TIME(stop);
    ns_ifft = CALC(start, stop);
    printf("NTT %u: %8lld - %8lld\n", logn, ns_fft, ns_ifft);
}

int main()
{
    test_FFT();
    test_NTT();

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
