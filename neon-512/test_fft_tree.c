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


/* see inner.h */
void
ifft(fpr *f, unsigned logn) {
    /*
     * Inverse FFT algorithm in bit-reversal order uses the following
     * iterative algorithm:
     *
     *   t = 1
     *   for m = N; m > 1; m /= 2:
     *       hm = m/2
     *       dt = t*2
     *       for i1 = 0; i1 < hm; i1 ++:
     *           j1 = i1 * dt
     *           s = iGM[hm + i1]
     *           for j = j1; j < (j1 + t); j ++:
     *               x = f[j]
     *               y = f[j + t]
     *               f[j] = x + y
     *               f[j + t] = s * (x - y)
     *       t = dt
     *   for i1 = 0; i1 < N; i1 ++:
     *       f[i1] = f[i1] / N
     *
     * iGM[k] contains (1/w)^rev(k) for primitive root w = exp(i*pi/N)
     * (actually, iGM[k] = 1/GM[k] = conj(GM[k])).
     *
     * In the main loop (not counting the final division loop), in
     * all iterations except the last, the first and second half of f[]
     * (as an array of complex numbers) are separate. In our chosen
     * representation, we do not keep the second half.
     *
     * The last iteration recombines the recomputed half with the
     * implicit half, and should yield only real numbers since the
     * target polynomial is real; moreover, s = i at that step.
     * Thus, when considering x and y:
     *    y = conj(x) since the final f[j] must be real
     *    Therefore, f[j] is filled with 2*Re(x), and f[j + t] is
     *    filled with 2*Im(x).
     * But we already have Re(x) and Im(x) in array slots j and j+t
     * in our chosen representation. That last iteration is thus a
     * simple doubling of the values in all the array.
     *
     * We make the last iteration a no-op by tweaking the final
     * division into a division by N/2, not N.
     */
    size_t u, falcon_n, hn, t, m;

    falcon_n = (size_t)1 << logn;
    t = 1;
    m = falcon_n;
    hn = falcon_n >> 1;
    // for (u = logn; u > 1; u --) {
    for (u = logn; u > 1; u --) {
        size_t hm, dt, i1, j1;

        hm = m >> 1;
        dt = t << 1;
        for (i1 = 0, j1 = 0; j1 < hn; i1 ++, j1 += dt) {
            size_t j, j2;

            j2 = j1 + t;
            fpr s_re, s_im;
            int tmp_re, tmp_im;
            tmp_re = ((hm + i1) << 1) + 0;
            tmp_im = ((hm + i1) << 1) + 1;

            s_re = fpr_gm_tab[tmp_re];
            s_im = fpr_gm_tab[tmp_im];
            for (j = j1; j < j2; j ++) {
                fpr x_re, x_im, y_re, y_im;

                x_re = f[j];
                x_im = f[j + hn];
                y_re = f[j + t];
                y_im = f[j + t + hn];
#if DEBUG == 1
                int i_xre, i_yre, i_xim, i_yim, i_sre, i_sim;
                i_xre = j;
                i_yre = j + t;
                i_xim = j + hn;
                i_yim = j + hn + t;
                i_sre = tmp_re;
                i_sim = tmp_im;
                (f[j]) = fpr_add(x_re, y_re);
                (f[j + hn]) = fpr_add(x_im, y_im);

                fpr fpct_d_re, fpct_d_im;
                fpr v1, v2; 
                v1 = fpr_sub(x_re, y_re);
                v2 = fpr_sub(x_im, y_im);

                if (u  < 0)
                {
                    printf("y_re: %d = (%d - %d) * %d + (%d - %d) * %d \n", i_yre, i_xim, i_yim, i_sim, i_xre, i_yre, i_sre);
                    printf("y_im: %d = (%d - %d) * %d - (%d - %d) * %d \n", i_yim, i_xim, i_yim, i_sre, i_xre, i_yre, i_sim);
                    printf("x_re: %d = %d + %d\n", i_xre, i_xre, i_yre);
                    printf("x_im: %d = %d + %d\n", i_xim, i_xim, i_yim);
                }
                fpct_d_re = fpr_add(fpr_mul(v1, s_re), fpr_mul(v2, s_im));
                fpct_d_im = fpr_sub(fpr_mul(v2, s_re), fpr_mul(v1, s_im));
                (f[j + t]) = fpct_d_re;
                (f[j + t + hn]) = fpct_d_im;
                
                // printf("x_re, y_re: %d, %d\t", j, j + t);
                // printf("%d, %d * (%d, %d)\n", j+t, j+t+hn, tmp_re, tmp_im);
#else 
                FPC_ADD(f[j], f[j + hn],
                        x_re, x_im, y_re, y_im);
                FPC_SUB(x_re, x_im, x_re, x_im, y_re, y_im);

                FPC_MUL(f[j + t], f[j + t + hn],
                x_re, x_im, s_re, s_im);
#endif

            }

        }
        t = dt;
        m = hm;
#if DEBUG == 1
        if (u  < 0)
        {
            printf("========%d\n", u);
            for (int i =0; i < 1 << logn; i++)
            {
                printf("%f, ", f[i]);
            }
            // for (int i = hn; i < 16 + 16; i++)
            // {
            //     printf("%f, ", f[i]);
            // }
            printf("\n");
        }
        printf("\n");
#endif
    }

    /*
     * Last iteration is a no-op, provided that we divide by N/2
     * instead of N. We need to make a special case for logn = 0.
     */

    if (logn > 0) {
        fpr ni;

        ni = fpr_p2_tab[logn];
        for (u = 0; u < falcon_n; u ++) {
            f[u] = fpr_mul(f[u], ni);
        }
    }
#if DEBUG == 1
    // printf("========%d\n", u);
    // for (int i =0; i < 8; i++)
    // {
    //     printf("%f, ", f[i]);
    // }
    // printf("\n");
#endif

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

    for (int kkk = 0; kkk < 1; kkk++)
    for (unsigned int logn = 1; logn < 11; logn += 1)
    {
        for (unsigned int j = 0; j < 1 << logn; j++)
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
    printf("Finish Testing FFT\n");

    for (int kkk = 0; kkk < 1; kkk++)
    for (unsigned int logn = 8; logn < 11; logn += 1)
    {
        printf("logn %u\n", logn);
        for (unsigned int j = 0; j < 1 << logn; j++)
        {
            tmp = fRand(-512, 512);
            f_gold[j] = tmp;
            f_test[j] = tmp;
            f_logn[j] = tmp;
        }
        if (logn < 0)
        {
            print_array(f_gold, 1 << logn, "before f_gold", 1);
        }

        ifft(f_gold, logn);
        PQCLEAN_FALCON512_CLEAN_iFFT_original(f_test, logn);
        Zf(iFFT_logn)(f_logn, logn);

        // print_array(f_logn, 1 << logn, "logn", 1);
        if (logn > 6)
        {
            print_array(f_gold, 1 << logn, "after f_gold", 1);
            print_array(f_logn, 1 << logn, "after f_logn", 1);
        }

        ret |= compare(f_test, f_gold, 1 << logn, "test vs gold");
        ret |= compare(f_gold, f_logn, 1 << logn, "test vs logn");
        if (ret)
        {
            printf("ERR logn = %u\n", logn);
            return 1;
        }
        printf("OK logn = %u\n", logn);
    }
    printf("Finish Testing iFFT\n");

    free(f_test);
    free(f_gold);
    free(f_logn);

    return 0;
}