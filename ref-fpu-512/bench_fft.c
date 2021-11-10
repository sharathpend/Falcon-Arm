#include "inner.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include "util.h"


#define NTESTS 1000000
#define FALCON_N (1 << 10)
#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);;
// Result is nanosecond per call 
#define  CALC(start, stop) ((double) ((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / NTESTS;

int test_FFT(fpr *f, unsigned logn)
{
    struct timespec start, stop;
    long long ns_fft, ns_ifft;
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
    printf("FFT %u: %lld - %lld\n", logn, ns_fft, ns_ifft);
}


int main()
{
    fpr f[FALCON_N];
    for (int i = 0; i < FALCON_N; i++)
    {
        f[i].v = (double) i;
    }

    for (unsigned i = 0; i < 11; i++)
    {
        test_FFT(f, i);
    }

    return 0;
}