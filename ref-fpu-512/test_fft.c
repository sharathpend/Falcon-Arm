#include "inner.h"

#include <stdio.h>
#include "fft_consts.c"

void combine(fpr *out, fpr *in, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;

    int i_out = 0;
    fpr re, im;
    for (int i = 0; i < hn; i++)
    {
        re = in[i];
        im = in[i + hn];
        out[i_out++] = re;
        out[i_out++] = im;
    }
}

void split(fpr *out, fpr *in, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;

    int i_out = 0;
    fpr re, im;
    for (int i = 0; i < n; i += 2)
    {
        re = in[i];
        im = in[i + 1];
        out[i_out] = re;
        out[i_out + hn] = im;
        i_out += 1;
    }
}

void print_double(fpr *f, unsigned logn, const char *string)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;

    printf("%s:\n", string);
    for (int i = 0; i < n; i += 2)
    {
        printf("%.1f, %.1f, ", f[i].v, f[i + 1].v);
    }
    printf("\n");
}

int cmp_double(fpr *f, fpr *g, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;

    for (int i = 0; i < n; i++)
    {
        if (fabs(f[i].v - g[i].v) > 0.5)
        {
            printf("[%d]: %.2f != %.2f \n", i, f[i].v, g[i].v);
            printf("ERROR\n");
            return 1;
        }
    }
    return 0;
}

void fwd_FFT(fpr *f, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;
    unsigned len, start, j, k;
    fpr zeta_re, zeta_im, t_re, t_im, a_re, a_im,
        b_re, b_im;

    k = 0;
    for (len = hn; len > 1; len >>= 1)
    {
        for (start = 0; start < n; start = j + len)
        {
            zeta_re = my_fpr_tab[k];
            zeta_im = my_fpr_tab[k + 1];
            for (j = start; j < start + len; j += 2)
            {
                a_re = f[j];
                a_im = f[j + 1];
                b_re = f[j + len];
                b_im = f[j + len + 1];

                printf("(%4d, %4d) * (%4d, %4d)\n", j + len, j + len + 1,
                       k, k + 1);

                // printf("@(%        4d, %4d) - (%4d, %4d)\n", j, j + 1, j + len, j + len + 1);
                // printf("@(%4d, %4d) + (%4d, %4d)\n", j, j + hn, j + ht, j + ht + hn);

                FPC_MUL(t_re, t_im, b_re, b_im, zeta_re, zeta_im);
                printf("FPC_MUL(%.1f, %.1f, %.1f, %.1f, %.2f, %.2f);\n\n", t_re.v, t_im.v,
                       b_re.v, b_im.v, zeta_re.v, zeta_im.v);

                FPC_SUB(f[j + len], f[j + len + 1], a_re, a_im, t_re, t_im);
                FPC_ADD(f[j], f[j + 1], a_re, a_im, t_re, t_im);
            }
            k += 2;
        }
        printf("---%4d\n", len);
    }
}

void inv_FFT(fpr *f, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;
    unsigned len, start, j, k;
    fpr zeta_re, zeta_im, t_re, t_im, a_re, a_im,
        b_re, b_im;

    switch (logn)
    {
        case 2: 
            k = hn;
            break;

        case 3: 
            k = hn + 2;
            break;

        case 4: 
            k = hn + 6; 
            break; 

        default:
            return;
    }

    for (len = 2; len < n; len <<= 1)
    {
        for (start = 0; start < n; start = j + len)
        {
            // Conjugate of zeta
            zeta_im = my_fpr_tab_inv[k - 1];
            zeta_re = my_fpr_tab_inv[k - 2];
            for (j = start; j < start + len; j += 2)
            {
                printf("len, j, start = %d, %d, %d\n", len, j, start);
                a_re = f[j];
                a_im = f[j + 1];
                b_re = f[j + len];
                b_im = f[j + len + 1];

                printf("(%4d, %4d) * (%4d, %4d)\n", j + len, j + len + 1,
                       k - 2, k - 1);

                FPC_ADD(f[j], f[j + 1], a_re, a_im, b_re, b_im);
                // printf("FPC_ADD(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f);\n", f[j].v, f[j + 1].v,
                //        a_re.v, a_im.v, b_re.v, b_im.v);

                FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
                // printf("FPC_SUB(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f);\n", t_re.v, t_im.v,
                //        a_re.v, a_im.v, b_re.v, b_im.v);

                FPC_MUL_CONJ(f[j + len], f[j + len + 1], t_re, t_im, zeta_re, zeta_im);
                // printf("FPC_MUL_CONJ(%.1f, %.1f, %.1f, %.1f, %.2f, %.2f);\n", f[j + len].v, f[j + len + 1].v,
                    //    t_re.v, t_im.v, zeta_re.v, zeta_im.v);
            }
            k -= 2;
            printf("---%4d\n", len);
        }
    }

    for (j = 0; j < n; j++)
    {
        f[j].v = f[j].v * fpr_p2_tab[logn].v;
    }
}

int test_fft_ifft(unsigned logn)
{
    fpr f[1024], g[1024], tmp[1024];
    for (int i = 0; i < 1024; i++)
    {
        f[i].v = (double_t)i;
    }
    // print_double(f, 10, "f");
    // print_double(g, 10, "g");
    combine(g, f, logn);

    // printf("=====\n");
    // Zf(FFT)(f, logn);
    // printf("=====\n");

    Zf(iFFT)(f, logn);
    printf("=====\n");

    // printf("-----\n");
    // fwd_FFT(g, logn);
    // printf("-----\n");
    inv_FFT(g, logn);
    printf("-----\n");

    split(tmp, g, logn);

    print_double(f, logn, "f");
    print_double(tmp, logn, "g");

    return cmp_double(f, tmp, logn);
}

int main()
{
    fpr f[1024], g[1024], tmp[1024];
    for (int i = 0; i < 1024; i++)
    {
        f[i].v = (double_t)i;
    }
    // print_double(f, 10, "f");
    // print_double(g, 10, "g");

    // if (test_fft_ifft(2))
    // {
    //     return 1;
    // }
    // printf("\nFINISH LOGN = 2\n");

    // if (test_fft_ifft(3))
    // {
    //     return 1;
    // }
    // printf("\nFINISH LOGN = 3\n");

    if (test_fft_ifft(4))
    {
        return 1;
    }
    printf("\nFINISH LOGN = 4\n");

    return 0;
}
