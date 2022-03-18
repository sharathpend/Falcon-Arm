#include "inner.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stddef.h>

#define Q 12289

#define NTESTS 1000000
#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);;
// Result is nanosecond per call 
#define  CALC(start, stop) ((double) ((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / NTESTS;



void test_FFT(unsigned logn)
{
    struct timespec start, stop;
    long long ns_fft, ns_ifft;

    const unsigned n = 1 << logn;

    fpr f[n];
    for (int i = 0; i < n; i++)
    {
        f[i] = fpr_of(i);
    }
    /* =================================== */
    
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        Zf(FFT)(f, logn);
    }
    TIME(stop);
    ns_fft = CALC(start, stop);
    /* =================================== */
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        Zf(iFFT)(f, logn);
    }
    /* =================================== */
    TIME(stop);
    ns_ifft = CALC(start, stop);
    printf("FFT %u: %8lld - %8lld\n", logn, ns_fft, ns_ifft);
}

void test_NTT(unsigned logn)
{
    struct timespec start, stop;
    long long ns_fft, ns_ifft;

    const unsigned n = 1 << logn;

    int16_t a[n];
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % Q;
    }

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        mq_NTT(a, logn);
    }
    TIME(stop);
    ns_fft = CALC(start, stop);
    /* =================================== */
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        mq_iNTT(a, logn);
    }
    /* =================================== */
    TIME(stop);
    ns_ifft = CALC(start, stop);
    printf("NTT %u: %8lld - %8lld\n", logn, ns_fft, ns_ifft);
}

int main()
{
    test_FFT(9);
    test_FFT(10);
    test_NTT(9);
    test_NTT(10);

    return 0;
}

/* 
 * Result in nanosection
 * FFT          - iFFT
 *  9: 8925     - 9825
 * 10: 19091    - 21477
 * 
 * NTT          - iNTT
 *  9: 15585    - 14865
 * 10: 33457    - 31627
 */
