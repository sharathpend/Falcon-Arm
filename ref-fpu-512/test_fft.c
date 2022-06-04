#include "inner.h"

#include <stdio.h>
#include "fft_consts.c"

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

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
        if (fabs(f[i].v - g[i].v) > 0.000001)
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
    fpr zeta_re, zeta_im, t_re, t_im, a_re, a_im, b_re, b_im;

    /*
     * We read the twiddle table in forward order
     */
    int level = 1;
    const fpr *fpr_tab = NULL;
    const fpr *table[] = {
        fpr_tab_log2,
        fpr_tab_log3,
        fpr_tab_log4,
        fpr_tab_log5,
        fpr_tab_log6,
        fpr_tab_log7,
        fpr_tab_log8,
        fpr_tab_log9,
        fpr_tab_log10};

    fpr_tab = table[0];
    zeta_re = fpr_tab[0];
    zeta_im = fpr_tab[1];
    for (j = 0; j < hn; j += 2)
    {
        a_re = f[j];
        a_im = f[j + 1];
        b_re = f[j + hn];
        b_im = f[j + hn + 1];

        FPC_MUL(t_re, t_im, b_re, b_im, zeta_re, zeta_im);
        FPC_SUB(f[j + hn], f[j + hn + 1], a_re, a_im, t_re, t_im);
        FPC_ADD(f[j], f[j + 1], a_re, a_im, t_re, t_im);
    }

    for (len = hn / 2; len > 1; len >>= 1)
    {
        fpr_tab = table[level++];
        k = 0;
        for (start = 0; start < n; start = j + len)
        {
            zeta_re = fpr_tab[k];
            zeta_im = fpr_tab[k + 1];
            k += 2;

            for (j = start; j < start + len; j += 2)
            {
                a_re = f[j];
                a_im = f[j + 1];
                b_re = f[j + len];
                b_im = f[j + len + 1];

                FPC_MUL(t_re, t_im, b_re, b_im, zeta_re, zeta_im);
                FPC_SUB(f[j + len], f[j + len + 1], a_re, a_im, t_re, t_im);
                FPC_ADD(f[j], f[j + 1], a_re, a_im, t_re, t_im);
            }

            start = j + len;

            for (j = start; j < start + len; j += 2)
            {
                a_re = f[j];
                a_im = f[j + 1];
                b_re = f[j + len];
                b_im = f[j + len + 1];

                FPC_MUL(t_re, t_im, b_re, b_im, zeta_re, zeta_im);
                FPC_SUBJ(f[j + len], f[j + len + 1], a_re, a_im, t_re, t_im);
                FPC_ADDJ(f[j], f[j + 1], a_re, a_im, t_re, t_im);
            }
        }
    }
}

void inv_FFT(fpr *f, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;
    unsigned len, start, j, k;
    fpr zeta_re, zeta_im, t_re, t_im, a_re, a_im, b_re, b_im;

    /*
     * This time we read the table in reverse order,
     * so the pointer point to the end of the table
     */
    int level = logn - 2;
    const fpr *fpr_tab_inv = NULL;
    const fpr *table[] = {
        fpr_tab_log2,
        fpr_tab_log3,
        fpr_tab_log4,
        fpr_tab_log5,
        fpr_tab_log6,
        fpr_tab_log7,
        fpr_tab_log8,
        fpr_tab_log9,
        fpr_tab_log10,
    };

    for (len = 2; len < hn; len <<= 1)
    {
        fpr_tab_inv = table[level--];
        k = 0;

        for (start = 0; start < n; start = j + len)
        {
            // Conjugate of zeta is embeded in MUL
            zeta_re = fpr_tab_inv[k];
            zeta_im = fpr_tab_inv[k + 1];
            k += 2;

            for (j = start; j < start + len; j += 2)
            {
                a_re = f[j];
                a_im = f[j + 1];
                b_re = f[j + len];
                b_im = f[j + len + 1];

                FPC_ADD(f[j], f[j + 1], a_re, a_im, b_re, b_im);
                FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
                FPC_MUL_CONJ(f[j + len], f[j + len + 1], t_re, t_im, zeta_re, zeta_im);
            }

            start = j + len;

            for (j = start; j < start + len; j += 2)
            {
                a_re = f[j];
                a_im = f[j + 1];
                b_re = f[j + len];
                b_im = f[j + len + 1];

                /*
                 * Notice we swap the (a - b) to (b - a) in FPC_SUB
                 */
                FPC_ADD(f[j], f[j + 1], a_re, a_im, b_re, b_im);
                FPC_SUB(t_re, t_im, b_re, b_im, a_re, a_im);
                FPC_MUL_CONJ_J_m(f[j + len], f[j + len + 1], t_re, t_im, zeta_re, zeta_im);
            }
        }
    }

    fpr_tab_inv = table[0];
    zeta_re.v = fpr_tab_inv[0].v * fpr_p2_tab[logn].v;
    zeta_im.v = fpr_tab_inv[1].v * fpr_p2_tab[logn].v;

    for (j = 0; j < hn; j += 2)
    {
        a_re = f[j];
        a_im = f[j + 1];
        b_re = f[j + hn];
        b_im = f[j + hn + 1];

        FPC_ADD(f[j], f[j + 1], a_re, a_im, b_re, b_im);
        FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
        FPC_MUL_CONJ(f[j + hn], f[j + hn + 1], t_re, t_im, zeta_re, zeta_im);

        f[j].v *= fpr_p2_tab[logn].v;
        f[j + 1].v *= fpr_p2_tab[logn].v;
    }
}

void split_fwd_FFT(fpr *f, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;
    const unsigned ht = n >> 2;
    unsigned len, start, j, k;
    fpr zeta_re, zeta_im, t_re, t_im, a_re, a_im, b_re, b_im;

    /*
     * We read the twiddle table in forward order
     */
    int level = 1;
    const fpr *fpr_tab = NULL;
    const fpr *table[] = {
        fpr_tab_log2,
        fpr_tab_log3,
        fpr_tab_log4,
        fpr_tab_log5,
        fpr_tab_log6,
        fpr_tab_log7,
        fpr_tab_log8,
        fpr_tab_log9,
        fpr_tab_log10};

    fpr_tab = table[0];
    zeta_re = fpr_tab[0];
    zeta_im = fpr_tab[1];

    for (j = 0; j < ht; j += 1)
    {
        a_re = f[j];
        a_im = f[j + hn];
        b_re = f[j + ht];
        b_im = f[j + ht + hn];

        FPC_MUL(t_re, t_im, b_re, b_im, zeta_re, zeta_im);
        FPC_SUB(f[j + ht], f[j + ht + hn], a_re, a_im, t_re, t_im);
        FPC_ADD(f[j], f[j + hn], a_re, a_im, t_re, t_im);
    }

    for (len = ht / 2; len > 0; len >>= 1)
    {
        fpr_tab = table[level++];
        k = 0;
        for (start = 0; start < hn; start = j + len)
        {
            zeta_re = fpr_tab[k];
            zeta_im = fpr_tab[k + 1];
            k += 2;
            for (j = start; j < start + len; j += 1)
            {
                a_re = f[j];
                a_im = f[j + hn];
                b_re = f[j + len];
                b_im = f[j + len + hn];

                FPC_MUL(t_re, t_im, b_re, b_im, zeta_re, zeta_im);
                FPC_SUB(f[j + len], f[j + len + hn], a_re, a_im, t_re, t_im);
                FPC_ADD(f[j], f[j + hn], a_re, a_im, t_re, t_im);
            }

            start = j + len;

            for (j = start; j < start + len; j += 1)
            {
                a_re = f[j];
                a_im = f[j + hn];
                b_re = f[j + len];
                b_im = f[j + len + hn];

                FPC_MUL(t_re, t_im, b_re, b_im, zeta_re, zeta_im);
                FPC_SUBJ(f[j + len], f[j + len + hn], a_re, a_im, t_re, t_im);
                FPC_ADDJ(f[j], f[j + hn], a_re, a_im, t_re, t_im);
            }
        }
    }
}

void split_inv_FFT(fpr *f, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;
    const unsigned ht = n >> 2;
    unsigned len, start, j, k;
    fpr zeta_re, zeta_im, t_re, t_im, a_re, a_im, b_re, b_im;

    /*
     * This time we read the table in reverse order,
     * so the pointer point to the end of the table
     */
    int level = logn - 2;
    const fpr *fpr_tab_inv = NULL;
    const fpr *table[] = {
        fpr_tab_log2,
        fpr_tab_log3,
        fpr_tab_log4,
        fpr_tab_log5,
        fpr_tab_log6,
        fpr_tab_log7,
        fpr_tab_log8,
        fpr_tab_log9,
        fpr_tab_log10,
    };

    for (len = 1; len < ht; len <<= 1)
    {
        fpr_tab_inv = table[level--];
        k = 0;

        for (start = 0; start < hn; start = j + len)
        {
            // Conjugate of zeta is embeded in MUL
            zeta_re = fpr_tab_inv[k];
            zeta_im = fpr_tab_inv[k + 1];
            k += 2;

            for (j = start; j < start + len; j += 1)
            {
                a_re = f[j];
                a_im = f[j + hn];
                b_re = f[j + len];
                b_im = f[j + len + hn];

                FPC_ADD(f[j], f[j + hn], a_re, a_im, b_re, b_im);
                FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
                FPC_MUL_CONJ(f[j + len], f[j + len + hn], t_re, t_im, zeta_re, zeta_im);
            }

            start = j + len;

            for (j = start; j < start + len; j += 1)
            {
                a_re = f[j];
                a_im = f[j + hn];
                b_re = f[j + len];
                b_im = f[j + len + hn];

                /*
                 * Notice we swap the (a - b) to (b - a) in FPC_SUB
                 */
                FPC_ADD(f[j], f[j + hn], a_re, a_im, b_re, b_im);
                FPC_SUB(t_re, t_im, b_re, b_im, a_re, a_im);
                FPC_MUL_CONJ_J_m(f[j + len], f[j + len + hn], t_re, t_im, zeta_re, zeta_im);
            }
        }
    }

    fpr_tab_inv = table[0];
    zeta_re.v = fpr_tab_inv[0].v * fpr_p2_tab[logn].v;
    zeta_im.v = fpr_tab_inv[1].v * fpr_p2_tab[logn].v;

    for (j = 0; j < ht; j += 1)
    {
        a_re = f[j];
        a_im = f[j + hn];
        b_re = f[j + ht];
        b_im = f[j + ht + hn];

        FPC_ADD(f[j], f[j + hn], a_re, a_im, b_re, b_im);
        FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
        FPC_MUL_CONJ(f[j + ht], f[j + ht + hn], t_re, t_im, zeta_re, zeta_im);

        f[j].v *= fpr_p2_tab[logn].v;
        f[j + hn].v *= fpr_p2_tab[logn].v;
    }
}

int test_fft_ifft(unsigned logn, unsigned tests)
{
    fpr f[1024], g[1024], tmp[1024];
    for (int j = 0; j < tests; j++)
    {
        for (int i = 0; i < 1024; i++)
        {
            f[i].v = drand(-12289.0, 12289);
        }
        combine(g, f, logn);

        Zf(FFT)(f, logn);
        Zf(iFFT)(f, logn);

        fwd_FFT(g, logn);
        inv_FFT(g, logn);
        split(tmp, g, logn);

        // print_double(f, logn, "f");
        // print_double(tmp, logn, "g");

        if (cmp_double(f, tmp, logn))
        {
            return 1;
        }
    }
    return 0;
}

int test_split_fft_ifft(unsigned logn, unsigned tests)
{
    fpr f[1024], g[1024];
    for (int j = 0; j < tests; j++)
    {
        for (int i = 0; i < 1024; i++)
        {
            f[i].v = drand(-12289.0, 12289);
            // f[i].v = (double_t) i;
            g[i].v = f[i].v;
        }
        Zf(FFT)(f, logn);
        Zf(iFFT)(f, logn);

        split_fwd_FFT(g, logn);
        split_inv_FFT(g, logn);

        // print_double(f, logn, "f");
        // print_double(g, logn, "g");

        if (cmp_double(f, g, logn))
        {
            return 1;
        }
    }
    return 0;
}

#define TESTS 10000

int main(void)
{
    printf("\ntest_fft_ifft: ");
    for (int logn = 2; logn < 11; logn++)
    {
        if (test_fft_ifft(logn, TESTS))
        {
            printf("Error at LOGN = %d\n", logn);
            return 1;
        }
    }
    printf("OK\n");

    printf("\ntest_split_fft_ifft: ");
    for (int logn = 2; logn < 11; logn++)
    {
        if (test_split_fft_ifft(logn, TESTS))
        {
            printf("Error at LOGN = %d\n", logn);
            return 1;
        }
    }
    printf("OK\n");

    return 0;
}
