#include <stdio.h>
#include "inner.h"
#include "fpr.h"

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
    fpr *fpr_tab = NULL;

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
    fpr *fpr_tab_inv = NULL;

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

void split_fwd_FFT_v0(fpr *f, unsigned logn)
{
    const unsigned n = 1 << logn;
    const unsigned hn = n >> 1;
    const unsigned ht = n >> 2;
    unsigned len, start, j, k;
    fpr zeta_re, zeta_im, t_re, t_im, a_re, a_im, b_re, b_im;

    /*
     * We read the twiddle table in forward order
     */
    int level = 0;
    fpr *fpr_tab = NULL;

    for (len = ht; len > 0; len >>= 1)
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
            if (start >= hn)
                break;

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

void split_inv_FFT_v0(fpr *f, unsigned logn)
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
    fpr *fpr_tab_inv = NULL;

    for (len = 1; len < hn; len <<= 1)
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
            if (start >= hn)
                break;

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

    for (j = 0; j < hn; j += 1)
    {
        f[j] *= fpr_p2_tab[logn];
        f[j + hn] *= fpr_p2_tab[logn];
    }
}

int test_fft_ifft(unsigned logn, unsigned tests)
{
    fpr f[1024], g[1024], tmp[1024];
    const unsigned n = 1 << logn;
    for (int j = 0; j < tests; j++)
    {
        for (int i = 0; i < n; i++)
        {
            f[i].v = drand(-12289.0, 12289);
        }
        combine(g, f, logn);

        Zf(FFT)(f, logn);
        Zf(iFFT)(f, logn);

        fwd_FFT(g, logn);
        inv_FFT(g, logn);
        split(tmp, g, logn);

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
    const unsigned n = 1 << logn;
    for (int j = 0; j < tests; j++)
    {
        for (int i = 0; i < n; i++)
        {
            f[i].v = drand(-12289.0, 12289);
            g[i].v = f[i].v;
        }
        Zf(FFT)(f, logn);
        Zf(FFT_ref)(g, logn);
        if (cmp_double(f, g, logn))
        {
            return 1;
        }

        Zf(iFFT)(f, logn);
        Zf(iFFT_ref)(g, logn);
        if (cmp_double(f, g, logn))
        {
            return 1;
        }
    }
    return 0;
}

int test_variant_fft(unsigned logn, unsigned tests)
{
    fpr f[1024], g[1024];
    const unsigned n = 1 << logn;
    for (int j = 0; j < tests; j++)
    {
        for (int i = 0; i < n; i++)
        {
            f[i] = drand(-12289.0, 12289);
            g[i] = f[i];
        }

        Zf(FFT)(f, logn);
        split_fwd_FFT_v0(g, logn);
        if (cmp_double(f, g, logn))
        {
            return 1;
        }

        Zf(iFFT)(f, logn);
        split_inv_FFT_v0(g, logn);
        if (cmp_double(f, g, logn))
        {
            return 1;
        }
    }
    return 0;
}

int test_split_merge_function(unsigned logn, unsigned tests)
{
    fpr f[1024] = {0}, f0[1024] = {0}, f1[1024] = {0};
    fpr g[1024] = {0}, g0[1024] = {0}, g1[1024] = {0};
    const unsigned n = 1 << logn;
    for (int j = 0; j < tests; j++)
    {
        for (int i = 0; i < n; i++)
        {
            f[i] = drand(-12289.0, 12289);
        }
        Zf(poly_split_fft)(f0, f1, f, logn);
        Zf(poly_split_fft_ref)(g0, g1, f, logn);

        if (cmp_double(g0, f0, logn))
        {
            print_double(f0, logn - 1, "f0");
            print_double(g0, logn - 1, "g0");
            printf("f0\n");
            return 1;
        }
        if (cmp_double(g1, f1, logn))
        {
            print_double(f1, logn - 1, "f1");
            print_double(g1, logn - 1, "g1");
            printf("f1\n");
            return 1;
        }

        Zf(poly_merge_fft)(f, f0, f1, logn);
        Zf(poly_merge_fft_ref)(g, g0, g1, logn);

        if (cmp_double(f, g, logn))
        {
            print_double(f, logn, "f");
            print_double(g, logn, "g");
            printf("f != g\n");
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

    printf("\ntest_variant_fft: ");
    for (int logn = 2; logn < 11; logn++)
    {
        if (test_variant_fft(logn, TESTS))
        {
            printf("Error at LOGN = %d\n", logn);
            return 1;
        }
    }
    printf("OK\n");

    printf("\ntest_split_fft_ifft: \n");
    for (int logn = 2; logn < 11; logn++)
    {
        if (test_split_fft_ifft(logn, TESTS))
        {
            printf("Error at LOGN = %d\n", logn);
            return 1;
        }
    }
    printf("OK\n");

    printf("\ntest_split_merge_function: \n");
    for (int logn = 2; logn < 11; logn++)
    {
        if (test_split_merge_function(logn, TESTS))
        {
            printf("Error at LOGN = %d\n", logn);
            return 1;
        }
    }
    printf("OK\n");

    return 0;
}
