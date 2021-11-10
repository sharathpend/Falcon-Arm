#include "inner.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include "util.h"


#define NTESTS 1000000
#define FALCON_N (1 << 12)
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
        f[i] = (double) i;
    }

    for (unsigned i = 0; i < 11; i++)
    {
        test_FFT(f, i);
    }

    return 0;
}

/* 
 * Result in nanosection
 * logn : FFT - iFFT GCC FMA NEON -O3 
 *  0: 32       - 25
 *  1: 9        - 10
 *  2: 16       - 17
 *  3: 36       - 37
 *  4: 64       - 65
 *  5: 212      - 183
 *  6: 530      - 480
 *  7: 1139     - 1014
 *  8: 2721     - 2482
 *  9: 5752     - 5189
 * 10: 13419    - 12382
 * 
 * logn: FFT - iFFT GCC REF -O3 
 *  0: 14       - 10
 *  1: 7        - 14
 *  2: 40       - 58
 *  3: 95       - 101
 *  4: 212      - 222
 *  5: 459      - 480
 *  6: 1000     - 1044
 *  7: 2159     - 2263
 *  8: 4616     - 4812
 *  9: 9897     - 10284
 * 10: 22036    - 21886
 */