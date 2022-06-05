/*
 * High-speed FFT code for arbitrary `logn`.
 *
 * =============================================================================
 * Copyright (c) 2022 by Cryptographic Engineering Research Group (CERG)
 * ECE Department, George Mason University
 * Fairfax, VA, U.S.A.
 * Author: Duc Tri Nguyen
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 * @author   Duc Tri Nguyen <dnguye69@gmu.edu>
 */

#include "inner.h"
#include "fpr.h"

/*
 * Addition of two complex numbers (d = a + b).
 */
#define FPC_ADD(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re.v = a_re.v + b_re.v;                       \
    d_im.v = a_im.v + b_im.v;

/*
 * Addition of two complex numbers (d = a + jb).
 */
#define FPC_ADDJ(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re.v = a_re.v - b_im.v;                        \
    d_im.v = a_im.v + b_re.v;
/*
 * Subtraction of two complex numbers (d = a - b).
 */
#define FPC_SUB(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re.v = a_re.v - b_re.v;                       \
    d_im.v = a_im.v - b_im.v;

/*
 * Subtraction of two complex numbers (d = a - jb).
 */
#define FPC_SUBJ(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re.v = a_re.v + b_im.v;                        \
    d_im.v = a_im.v - b_re.v;

/*
 * Multplication of two complex numbers (d = a * b).
 */
#define FPC_MUL(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re.v = a_re.v * b_re.v - a_im.v * b_im.v;     \
    d_im.v = a_re.v * b_im.v + a_im.v * b_re.v;

/*
 * Multplication of two complex numbers (d = a * conj(b)).
 * a is swapped from: a_re|a_im to a_im|a_re
 * b is swapped from: b_re|b_im to b_im|b_re
 */
#define FPC_MUL_CONJ(d_re, d_im, a_im, a_re, b_im, b_re) \
    d_re.v = b_re.v * a_re.v + a_im.v * b_im.v;          \
    d_im.v = b_im.v * a_re.v - a_im.v * b_re.v;

/*
 * Multplication of two complex numbers (d = a * conj(jb)).
 */
#define FPC_MUL_CONJ_J(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re.v = a_im.v * b_re.v - b_im.v * a_re.v;            \
    d_im.v = -(a_im.v * b_im.v + b_re.v * a_re.v);

/*
 * Multplication of two complex numbers (d = a * - conj(jb)).
 * b is swapped from: b_re|b_im to b_im|b_re
 */
#define FPC_MUL_CONJ_J_m(d_re, d_im, a_re, a_im, b_im, b_re) \
    d_re.v = a_re.v * b_re.v - a_im.v * b_im.v;              \
    d_im.v = a_im.v * b_re.v + a_re.v * b_im.v;


void Zf(FFT)(fpr *f, unsigned logn)
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
    fpr *fpr_tab = fpr_table[0];
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
        fpr_tab = fpr_table[level++];
        k = 0;
        for (start = 0; start < hn; start = j + len)
        {
            zeta_re = fpr_tab[k++];
            zeta_im = fpr_tab[k++];
            
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


void Zf(iFFT)(fpr *f, unsigned logn)
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
    
    for (len = 1; len < ht; len <<= 1)
    {
        fpr_tab_inv = fpr_table[level--];
        k = 0;
        for (start = 0; start < hn; start = j + len)
        {
            // Conjugate of zeta is embeded in MUL
            zeta_re = fpr_tab_inv[k++];
            zeta_im = fpr_tab_inv[k++];

            for (j = start; j < start + len; j += 1)
            {
                a_re = f[j];
                a_im = f[j + hn];
                b_re = f[j + len];
                b_im = f[j + len + hn];

                FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
                FPC_ADD(f[j], f[j + hn], a_re, a_im, b_re, b_im);
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
                FPC_SUB(t_re, t_im, b_re, b_im, a_re, a_im);
                FPC_ADD(f[j], f[j + hn], a_re, a_im, b_re, b_im);
                FPC_MUL_CONJ_J_m(f[j + len], f[j + len + hn], t_re, t_im, zeta_re, zeta_im);
            }
        }
    }


    fpr_tab_inv = fpr_table[0];
    zeta_re = fpr_mul(fpr_tab_inv[0], fpr_p2_tab[logn]);
    zeta_im = fpr_mul(fpr_tab_inv[1], fpr_p2_tab[logn]);
    for (j = 0; j < ht; j += 1)
    {
        a_re = f[j];
        a_im = f[j + hn];
        b_re = f[j + ht];
        b_im = f[j + ht + hn];

        FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
        FPC_ADD(f[j], f[j + hn], a_re, a_im, b_re, b_im);
        FPC_MUL_CONJ(f[j + ht], f[j + ht + hn], t_re, t_im, zeta_re, zeta_im);

        f[j] = fpr_mul(f[j], fpr_p2_tab[logn]);
        f[j + hn] = fpr_mul(f[j + hn], fpr_p2_tab[logn]);
    }
}



void Zf(poly_split_fft)(fpr *restrict f0, fpr *restrict f1,
	                    const fpr *restrict f, unsigned logn)
{
    const unsigned n = 1 << logn; 
    const unsigned hn = n >> 1; 
    const unsigned ht = n >> 2; 

    const fpr *fpr_split = fpr_table[logn - 2];
    unsigned k = 0;

    f0[0] = f[0];
    f1[0] = f[hn];

    fpr a_re, a_im, b_re, b_im, t_re, t_im, v_re, v_im, s_re, s_im;
    for (unsigned j = 0; j < ht; j++)
    {
        unsigned j2 = j << 1; 
        a_re = f[j2];
        b_re = f[j2 + 1];
        a_im = f[j2 + hn];
        b_im = f[j2 + 1 + hn];

        s_re = fpr_half(fpr_split[k++]);
        s_im = fpr_half(fpr_split[k++]);

        FPC_ADD(v_re, v_im, a_re, a_im, b_re, b_im);
        f0[j] = fpr_half(v_re); 
        f0[j + ht] = fpr_half(v_im); 

        FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);
        FPC_MUL_CONJ(f1[j], f1[j + ht], t_re, t_im, s_re, s_im);
        
        // ===========
        j += 1; 
        if (j >= ht) break;

        j2 = j << 1; 
        
        a_re = f[j2];
        b_re = f[j2 + 1];
        a_im = f[j2 + hn];
        b_im = f[j2 + 1 + hn];

        FPC_ADD(v_re, v_im, a_re, a_im, b_re, b_im);
        f0[j] = fpr_half(v_re); 
        f0[j + ht] = fpr_half(v_im); 

        /*
        * Notice we swap the (a - b) to (b - a) in FPC_SUB
        */
        FPC_SUB(t_re, t_im, b_re, b_im, a_re, a_im);
        FPC_MUL_CONJ_J_m(f1[j], f1[j + ht], t_re, t_im, s_re, s_im);
    }
}


void Zf(poly_merge_fft)(fpr *restrict f, const fpr *restrict f0, 
                            const fpr *restrict f1, unsigned logn)
{
    const unsigned n = 1 << logn; 
    const unsigned hn = n >> 1; 
    const unsigned ht = n >> 2; 

    const fpr *fpr_merge = fpr_table[logn - 2];
    unsigned k = 0;

    fpr a_re, a_im, b_re, b_im, t_re, t_im, v_re, v_im, s_re, s_im;

    for (unsigned j = 0; j < ht; j+= 1)
    {
        unsigned j2 = j << 1;

        a_re = f0[j];
        a_im = f0[j + ht];
        b_re = f1[j];
        b_im = f1[j + ht];

        s_re = fpr_merge[k++];
        s_im = fpr_merge[k++];

        FPC_MUL(t_re, t_im, b_re, b_im, s_re, s_im);
        FPC_ADD(f[j2], f[j2 + hn], a_re, a_im, t_re, t_im);
        FPC_SUB(f[j2 + 1], f[j2 + 1 + hn], a_re, a_im, t_re, t_im);

        j += 1; 
        if (j >= ht) break; 

        j2 = j << 1; 

        a_re = f0[j];
        a_im = f0[j + ht];
        
        b_re = f1[j];
        b_im = f1[j + ht];

        FPC_MUL(t_re, t_im, b_re, b_im, s_re, s_im);
        FPC_ADDJ(f[j2], f[j2 + hn], a_re, a_im, t_re, t_im);
        FPC_SUBJ(f[j2 + 1], f[j2 + 1 + hn], a_re, a_im, t_re, t_im);
    }
}