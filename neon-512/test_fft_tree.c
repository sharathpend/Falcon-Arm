#include <arm_neon.h>
#include <stdio.h>
#include "inner.h"
#include "util.h"
#include "vfpr.h"

#define DEBUG 1

void fft(fpr *f, unsigned logn)
{

    unsigned u;
    size_t t, n, hn, m;

    n = (size_t)1 << logn;
    hn = n >> 1;
    t = hn;
    for (u = 1, m = 2; u < logn; u++, m <<= 1)
    {
        size_t ht, hm, i1, j1;

        ht = t >> 1;
        hm = m >> 1;
        for (i1 = 0, j1 = 0; i1 < hm; i1++, j1 += t)
        {
            size_t j, j2;

            j2 = j1 + ht;
            fpr s_re, s_im;
            int bla, blo;
            bla = ((m + i1) << 1) + 0;
            blo = ((m + i1) << 1) + 1;
            s_re = fpr_gm_tab[bla];
            s_im = fpr_gm_tab[blo];
            for (j = j1; j < j2; j++)
            {
                fpr x_re, x_im, y_re, y_im;

                x_re = f[j];
                x_im = f[j + hn];
                y_re = f[j + ht];
                y_im = f[j + ht + hn];
#if DEBUG == 1
                fpr v1_re, v1_im;
                fpr v2_re, v2_im;

                v1_re = x_re;
                v1_im = x_im;

                v2_re = fpr_sub(fpr_mul(y_re, s_re), fpr_mul(y_im, s_im));
                v2_im = fpr_add(fpr_mul(y_re, s_im), fpr_mul(y_im, s_re));

                x_re = fpr_add(v1_re, v2_re);
                x_im = fpr_add(v1_im, v2_im);
                (f[j]) = x_re;
                (f[j + hn]) = x_im;

                y_re = fpr_sub(v1_re, v2_re);
                y_im = fpr_sub(v1_im, v2_im);
                (f[j + ht]) = y_re;
                (f[j + ht + hn]) = y_im;

                if (logn > 100)
                {
                    // printf("rvg: %f | %f\n", v2_re, v2_im);
                    // printf("radd: %f | %f\n", x_re, x_im);
                    // printf("rsub: %f | %f\n\n", y_re, y_im);
                    // printf("level %u\n", u);
                    // printf("v_re: (%3d*%3d - %3d*%3d)\n", j + ht, bla, j + ht + hn, blo);
                    // printf("v_im: (%3d*%3d + %3d*%3d)\n", j + ht, blo, j + ht + hn, bla);
                    printf("x_re: %3d = %3d + (%3d*%3d - %3d*%3d)\n", j, j, j + ht, bla, j + ht + hn, blo);
                    printf("y_re: %3d = %3d - (%3d*%3d - %3d*%3d)\n", j + ht, j, j + ht, bla, j + ht + hn, blo);
                    printf("x_im: %3d = %3d + (%3d*%3d + %3d*%3d)\n", j + hn, j + hn, j + ht, blo, j + ht + hn, bla);
                    printf("y_im: %3d = %3d - (%3d*%3d + %3d*%3d)\n", j + ht + hn, j + hn, j + ht, blo, j + ht + hn, bla);
                    // printf("----\n");
                }
#else
                FPC_MUL(y_re, y_im, y_re, y_im, s_re, s_im);
                FPC_ADD(f[j], f[j + hn],
                        x_re, x_im, y_re, y_im);
                FPC_SUB(f[j + ht], f[j + ht + hn],
                        x_re, x_im, y_re, y_im);
#endif
            }
            // print_array(f, 1 << logn, "level 6", 1);
            // return;
        }
        t = ht;
    }
}

int main()
{

    fpr *f_gold = malloc(FALCON_N * sizeof(double));
    fpr *f_test = malloc(FALCON_N * sizeof(double));
    fpr *f_logn = malloc(FALCON_N * sizeof(double));
    fpr tmp;
    int size = 8;
    int ret = 0;

    for (int i = 0; i < FALCON_N; i++)
    {
        tmp = i;
        f_gold[i] = tmp;
        f_test[i] = tmp;
        f_logn[i] = tmp;
    }
    // print_array(f_fft, FALCON_N, "Before FFT", 1);
    // PQCLEAN_FALCON512_NEON_iFFT(f_fft);
    // PQCLEAN_FALCON512_NEON_FFT(f_fft);
    // print_layer(f_fft, 16, FALCON_N);
    // print_array(f_fft, FALCON_N, "FFT", 1);

    // PQCLEAN_FALCON512_NEON_poly_split_fft(f0_gold, f1_gold, f_gold, size);
    // PQCLEAN_FALCON512_NEON_poly_merge_fft(f_test, f0_gold, f1_gold, size);

    // print_array(f0_gold, 1 << (size - 1), "f0_gold", 1);
    // print_array(f1_gold, 1 << (size - 1), "f1_gold", 1);
    // print_array(f_test, 1 << size, "f_test", 1);

    // ret |= compare(f_gold, f_test, 1 << size, "Compare with original");

    while (1)
    for (unsigned int logn = 1; logn < 11; logn += 1)
    {
        for (unsigned int j = 0; j < FALCON_N; j++)
        {
            tmp = fRand(-512, 512);
            f_gold[j] = tmp;
            f_test[j] = tmp;
            f_logn[j] = tmp;
        }
        // if (logn < 3)
        // {
        //     print_array(f_gold, 1 << logn, "before f_gold", 1);
        // }

        fft(f_gold, logn);
        PQCLEAN_FALCON512_CLEAN_FFT_original(f_test, logn);
        Zf(FFT_logn)(f_logn, logn);

        // print_array(f_logn, 1 << logn, "logn", 1);
        if (logn > 20)
        {
            print_array(f_gold, 1 << logn, "after f_gold", 1);
            print_array(f_logn, 1 << logn, "after f_gold", 1);
        }

        ret |= compare(f_test, f_gold, 1 << logn, "");
        ret |= compare(f_gold, f_logn, 1 << logn, "");
        if (ret)
        {
            printf("logn = %u\n", logn);
            return 1;
        }

        // for (int j = 0; j < FALCON_N; j++)
        // {
        //     tmp = 0.0;
        //     f_gold[i] = tmp;
        //     f_test[i] = tmp;
        //     f0_gold[i] = tmp;
        // }
    }

    free(f_test);
    free(f_gold);
    free(f_logn);

    return 0;
}