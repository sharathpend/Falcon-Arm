#include "inner.h"

void PQCLEAN_FALCON512_NEON_iFFT(fpr *f)
{
    // Total: 32 = 16 + 8 + 8 register
    float64x2x4_t s_re_im, tmp;           // 8
    float64x2x4_t x_re, x_im, y_re, y_im; // 16
    float64x2x4_t v_re, v_im;             // 8
    float64x2x2_t s_tmp;                  // 2
    // Level 4, 5
    float64x2x2_t x_tmp, y_tmp;
    // Level 6, 7
    float64x2_t div_n;

    const unsigned int hn = FALCON_N >> 1;

    // Level 0, 1, 2, 3
    for (int j = 0; j < FALCON_N / 2; j += 16)
    {
        // Level 0
        // x_re = 0, 4 | 2, 6 | 8, 12 | 10, 14
        // y_re = 1, 5 | 3, 7 | 9, 13 | 11, 15
        // x_im = 256, 260 | 258, 262 | 264, 268 | 266, 270
        // y_im = 257, 261 | 259, 263 | 265, 269 | 267, 271

        // This assignment is free
        vload4(tmp, &f[j]);
        x_re.val[0] = tmp.val[0];
        x_re.val[1] = tmp.val[2];
        y_re.val[0] = tmp.val[1];
        y_re.val[1] = tmp.val[3];

        vload4(tmp, &f[j + 8]);
        x_re.val[2] = tmp.val[0];
        x_re.val[3] = tmp.val[2];
        y_re.val[2] = tmp.val[1];
        y_re.val[3] = tmp.val[3];

        vload4(tmp, &f[j + hn]);
        x_im.val[0] = tmp.val[0];
        x_im.val[1] = tmp.val[2];
        y_im.val[0] = tmp.val[1];
        y_im.val[1] = tmp.val[3];

        vload4(tmp, &f[j + hn + 8]);
        x_im.val[2] = tmp.val[0];
        x_im.val[3] = tmp.val[2];
        y_im.val[2] = tmp.val[1];
        y_im.val[3] = tmp.val[3];

        ////////
        // x - y
        ////////

        //  1,5 <= 0,4  -  1,5 |   3,7 <=   2,6 -  3,7
        // 9,13 <= 8,12 - 9,13 | 11,15 <= 10,14 - 11,15
        vfsubx4(v_re, x_re, y_re);

        // 257,261 <= 256,260 - 257,261 | 259,263 <= 258,262 - 259,263
        // 265,269 <= 264,268 - 265,269 | 267,271 <= 266,270 - 267,271
        vfsubx4(v_im, x_im, y_im);

        ////////
        // x + y
        ////////

        // x_re: 0,4 | 2,6 | 8,12 | 10,14
        // 0,4  <= 0,4  +  1,5 |   2,6 <=   2,6 +   3,7
        // 8,12 <= 8,12 + 9,13 | 10,14 <= 10,14 + 11,15
        vfaddx4(x_re, x_re, y_re);

        // x_im: 256,260 | 258,262 | 264,268 | 266,270
        // 256,260 <= 256,260 + 257,261 | 258,262 <= 258,262 + 259,263
        // 264,268 <= 264,268 + 265,269 | 266,270 <= 266,270 + 267,271
        vfaddx4(x_im, x_im, y_im);

        // s * (x - y) = s*v = (s_re + i*s_im)(v_re + i*v_im)
        // y_re:  v_re*s_re + v_im*s_im
        // y_im: -v_re*s_im + v_im*s_re

        // s_re_im = 512 -> 519
        // s_re: 512 -> 518, step 2
        // s_im: 513 -> 519, step 2
        vload4(s_re_im, &fpr_gm_tab[FALCON_N + j]);

        // y_re: 1,5 | 3,7 | 9,13 | 11,15
        // y_re = v_im*s_im + v_re*s_re
        vfmul(tmp.val[0], v_im.val[0], s_re_im.val[1]);
        vfmul(tmp.val[1], v_im.val[1], s_re_im.val[3]);
        vfma(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0]);
        vfma(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2]);

        vfmul(tmp.val[0], v_im.val[0], s_re_im.val[0]);
        vfmul(tmp.val[1], v_im.val[1], s_re_im.val[2]);
        vfms(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[1]);
        vfms(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[3]);

        // y_im: 257,261 | 259,263 | 265,269 | 267,271
        // y_im = v_im*s_re - v_re*s_im
        vload4(s_re_im, &fpr_gm_tab[FALCON_N + j + 8]);

        vfmul(tmp.val[2], v_im.val[2], s_re_im.val[1]);
        vfmul(tmp.val[3], v_im.val[3], s_re_im.val[3]);
        vfma(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[0]);
        vfma(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2]);

        vfmul(tmp.val[2], v_im.val[2], s_re_im.val[0]);
        vfmul(tmp.val[3], v_im.val[3], s_re_im.val[2]);
        vfms(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1]);
        vfms(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[3]);

        // Level 1
        // x_re = 0,4 | 2,6 | 8,12 | 10,14
        // y_re = 1,5 | 3,7 | 9,13 | 11,15
        // x_im = 256,260 | 258,262 | 264,268 | 266,270
        // y_im = 257,261 | 259,263 | 265,269 | 267,271

        vload2(s_tmp, &fpr_gm_tab[(FALCON_N + j) >> 1]);
        s_re_im.val[0] = s_tmp.val[0];
        s_re_im.val[1] = s_tmp.val[1];
        vload2(s_tmp, &fpr_gm_tab[(FALCON_N + j + 8) >> 1]);
        s_re_im.val[2] = s_tmp.val[0];
        s_re_im.val[3] = s_tmp.val[1];

        ////////
        // x - y
        ////////

        // 0,4 - 2,6 | 8,12 - 10,14
        // 1,5 - 3,7 | 9,13 - 11,15
        vfsubx4_swap(v_re, x_re, y_re, 0, 1, 2, 3);
        // 256,260 - 258,262 | 264,268 - 266,270
        // 257,261 - 259,263 | 265,269 - 267,271
        vfsubx4_swap(v_im, x_im, y_im, 0, 1, 2, 3);

        ////////
        // x + y
        ////////

        // x_re: 0,4 | 8,12 | 1,5 | 9,13
        // 0,4 <= 0,4 + 2,6 | 8,12 <= 8,12 + 10,14
        // 1,5 <= 1,5 + 3,7 | 9,13 <= 9,13 + 11,15
        vfaddx4_swap(x_re, x_re, y_re, 0, 1, 2, 3);

        // x_im: 256, 260 | 264, 268 | 257, 261 | 265, 269
        // 256,260 <= 256,260 + 258,262 | 264,268 <= 264,268 + 266,270
        // 257,261 <= 257,261 + 259,263 | 265,269 <= 265,269 + 267,271
        vfaddx4_swap(x_im, x_im, y_im, 0, 1, 2, 3);

        // Calculate y
        // v_im*s_im
        vfmul(tmp.val[0], v_im.val[0], s_re_im.val[1]);
        vfmul(tmp.val[1], v_im.val[1], s_re_im.val[3]);
        vfmul(tmp.val[2], v_im.val[2], s_re_im.val[1]);
        vfmul(tmp.val[3], v_im.val[3], s_re_im.val[3]);

        // v_im*s_im + v_re*s_re
        // y_re: 2,6 | 10,14 | 3,7 | 11,15
        vfma(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0]);
        vfma(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2]);
        vfma(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[0]);
        vfma(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2]);

        // v_im*s_re
        vfmul(tmp.val[0], v_im.val[0], s_re_im.val[0]);
        vfmul(tmp.val[1], v_im.val[1], s_re_im.val[2]);
        vfmul(tmp.val[2], v_im.val[2], s_re_im.val[0]);
        vfmul(tmp.val[3], v_im.val[3], s_re_im.val[2]);

        // v_im*s_re - v_re*s_im
        // y_im: 258,262 | 266,270 | 259,263 | 267,271
        vfms(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[1]);
        vfms(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[3]);
        vfms(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1]);
        vfms(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[3]);

        // Level 2
        // Before Transpose
        // x_re = 0,4 | 8,12 | 1,5 | 9,13
        // y_re = 2,6 | 10,14 | 3,7 | 11,15
        // x_im = 256,260 | 264,268 | 257,261 | 265,269
        // y_im = 258,262 | 266,270 | 259,263 | 267,271

        transpose(x_re, x_re, tmp, 0, 2, 0);
        transpose(x_re, x_re, tmp, 1, 3, 1);
        transpose(y_re, y_re, tmp, 0, 2, 2);
        transpose(y_re, y_re, tmp, 1, 3, 3);

        transpose(x_im, x_im, tmp, 0, 2, 0);
        transpose(x_im, x_im, tmp, 1, 3, 1);
        transpose(y_im, y_im, tmp, 0, 2, 2);
        transpose(y_im, y_im, tmp, 1, 3, 3);

        // After Transpose
        // x_re = 0,1 | 8,9 | 4,5 | 12,13
        // y_re = 2,3 | 10,11 | 6,7 | 14,15
        // x_im = 256,257 | 264,265 | 260,261 | 268,269
        // y_im = 258,259 | 266,267 | 262,263 | 270,271

        vload2(s_tmp, &fpr_gm_tab[(FALCON_N + j) >> 2]);
        s_re_im.val[0] = s_tmp.val[0];
        s_re_im.val[1] = s_tmp.val[1];

        ////////
        // x - y
        ////////

        // 0,1 - 4,5   | 2,3 - 6,7
        // 8,9 - 12,13 | 10,11 - 14,15
        vfsub(v_re.val[0], x_re.val[0], x_re.val[2]);
        vfsub(v_re.val[2], x_re.val[1], x_re.val[3]);
        vfsub(v_re.val[1], y_re.val[0], y_re.val[2]);
        vfsub(v_re.val[3], y_re.val[1], y_re.val[3]);

        // 256,257 - 260,261 | 258,259 - 262,263
        // 264,265 - 268,269 | 266,267 - 270,271
        vfsub(v_im.val[0], x_im.val[0], x_im.val[2]);
        vfsub(v_im.val[2], x_im.val[1], x_im.val[3]);
        vfsub(v_im.val[1], y_im.val[0], y_im.val[2]);
        vfsub(v_im.val[3], y_im.val[1], y_im.val[3]);

        ////////
        // x + y
        ////////

        // x_re: 0,1 | 2,3 |  8,9 | 10,11
        // 0,1 <= 0,1 + 4,5  | 2,3 <= 2,3 + 6,7
        // 8,9 <= 8,9 + 12,13| 10,11 <= 10,11 + 14,15
        vfadd(x_re.val[0], x_re.val[0], x_re.val[2]);
        vfadd(x_re.val[2], x_re.val[1], x_re.val[3]);
        vfadd(x_re.val[1], y_re.val[0], y_re.val[2]);
        vfadd(x_re.val[3], y_re.val[1], y_re.val[3]);

        // x_im: 256,257 | 258,259 | 264,265 | 266,267
        // 256,257 <= 256,257 + 260,261 | 258,259 <= 258,259 + 262,263
        // 264,265 <= 264,265 + 268,269 | 266,267 <= 266,267 + 270,271
        vfadd(x_im.val[0], x_im.val[0], x_im.val[2]);
        vfadd(x_im.val[2], x_im.val[1], x_im.val[3]);
        vfadd(x_im.val[1], y_im.val[0], y_im.val[2]);
        vfadd(x_im.val[3], y_im.val[1], y_im.val[3]);

        // Calculate y
        // v_im*s_im
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[1], 0);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[1], 0);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[1], 1);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[1], 1);

        // v_im*s_im + v_re*s_re
        // y_re: 4,5 | 6,7 | 12,13 | 14,15
        vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 0);
        vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 0);
        vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[0], 1);
        vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[0], 1);

        // v_im*s_re
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 0);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 0);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[0], 1);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[0], 1);

        // v_im*s_re - v_re*s_im
        // y_im: 260,261 |  262,263 | 268,269 | 270,271
        vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[1], 0);
        vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[1], 0);
        vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1], 1);
        vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[1], 1);

        // Level 3
        // x_re: 0,1 | 2,3 | 8,9 | 10,11
        // y_re: 4,5 | 6,7 | 12,13 | 14,15
        // x_im: 256,257 | 258,259 | 264,265 | 266,267
        // y_im: 260,261 | 262,263 | 268,269 | 270,271

        // Load s_re_im
        vload(s_re_im.val[0], &fpr_gm_tab[(FALCON_N + j) >> 3]);
        ////////
        // x - y
        ////////

        // 0,1 -   8,9 | 2,3 - 10,11
        // 4,5 - 12,13 | 6,7 - 14,15
        vfsubx4_swap(v_re, x_re, y_re, 0, 2, 1, 3);

        // 256,257 - 264,265 | 258,259 - 266,267
        // 260,261 - 268,269 | 262,263 - 270,271
        vfsubx4_swap(v_im, x_im, y_im, 0, 2, 1, 3);

        ////////
        // x + y
        ////////

        // x_re: 0,1 | 2,3 | 4,5 | 6,7
        // 0,1 <= 0,1 +   8,9 | 2,3 <= 2,3 + 10,11
        // 4,5 <= 4,5 + 12,13 | 6,7 <= 6,7 + 14,15
        vfaddx4_swap(x_re, x_re, y_re, 0, 2, 1, 3);

        // x_im: 256,257 | 258,259 | 260,261 | 262,263
        // 256,257 <= 256,257 + 264,265 | 258,259 <= 258,259 + 266,267
        // 260,261 <= 260,261 + 268,269 | 262,263 <= 262,263 + 270,271
        vfaddx4_swap(x_im, x_im, y_im, 0, 2, 1, 3);

        // Calculate y
        // v_im*s_im
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 1);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 1);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[0], 1);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[0], 1);

        // v_im*s_im + v_re*s_re
        // y_re: 8,9 | 10,11 | 12,13 | 14,15
        vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 0);
        vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 0);
        vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[0], 0);
        vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[0], 0);

        // v_im*s_re
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 0);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 0);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[0], 0);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[0], 0);

        // v_im*s_re - v_re*s_im
        // y_im: 264,265 | 266,267 | 268,269 | 270,271
        vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 1);
        vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 1);
        vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[0], 1);
        vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[0], 1);

        // x_re: 0,1 | 2,3 | 4,5 | 6,7
        // y_re: 8,9 | 10,11 | 12,13 | 14,15
        // x_im: 256,257 | 258,259 | 260,261 | 262,263
        // y_im: 264,265 | 266,267 | 268,269 | 270,271
        vstorex4(&f[j], x_re);
        vstorex4(&f[j + 8], y_re);
        vstorex4(&f[j + hn], x_im);
        vstorex4(&f[j + hn + 8], y_im);
    }

    for (int i = 0; i < FALCON_N / 2; i += 1 << 6)
    {
        vloadx2(s_tmp, &fpr_gm_tab[(FALCON_N + i) >> 4]);
        s_re_im.val[0] = s_tmp.val[0];
        s_re_im.val[1] = s_tmp.val[1];
        vload(s_re_im.val[2], &fpr_gm_tab[(FALCON_N + i) >> 5]);

        for (int j = i; j < i + 16; j += 4)
        {
            // Layer 4
            // x_re: 0 ->3, 32->35
            // y_re: 16 -> 19, 48 -> 51
            // x_im: 256 -> 259, 288-> 291
            // y_im: 272 -> 275, 304 -> 307
            vloadx2(x_tmp, &f[j]);
            x_re.val[0] = x_tmp.val[0];
            x_re.val[1] = x_tmp.val[1];
            vloadx2(y_tmp, &f[j + 16]);
            y_re.val[0] = y_tmp.val[0];
            y_re.val[1] = y_tmp.val[1];

            vloadx2(x_tmp, &f[j + 32]);
            x_re.val[2] = x_tmp.val[0];
            x_re.val[3] = x_tmp.val[1];
            vloadx2(y_tmp, &f[j + 48]);
            y_re.val[2] = y_tmp.val[0];
            y_re.val[3] = y_tmp.val[1];

            vloadx2(x_tmp, &f[j + hn]);
            x_im.val[0] = x_tmp.val[0];
            x_im.val[1] = x_tmp.val[1];
            vloadx2(y_tmp, &f[j + hn + 16]);
            y_im.val[0] = y_tmp.val[0];
            y_im.val[1] = y_tmp.val[1];

            vloadx2(x_tmp, &f[j + hn + 32]);
            x_im.val[2] = x_tmp.val[0];
            x_im.val[3] = x_tmp.val[1];
            vloadx2(y_tmp, &f[j + hn + 48]);
            y_im.val[2] = y_tmp.val[0];
            y_im.val[3] = y_tmp.val[1];

            ////////
            // x - y
            ////////
            vfsubx4(v_re, x_re, y_re);
            vfsubx4(v_im, x_im, y_im);

            ////////
            // x + y
            ////////
            // x_re: 0->3, 32->35
            // x_im: 256->259, 288->291
            vfaddx4(x_re, x_re, y_re);
            vfaddx4(x_im, x_im, y_im);

            // Calculate y
            // v_im*s_im
            vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 1);
            vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 1);
            vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[1], 1);
            vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[1], 1);

            // v_im*s_im + v_re*s_re
            // y_re: 16->19, 48->51
            vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 0);
            vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 0);
            vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1], 0);
            vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[1], 0);

            // v_im*s_re
            vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 0);
            vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 0);
            vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[1], 0);
            vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[1], 0);

            // v_im*s_re - v_re*s_im
            // y_im: 272->275, 304->307
            vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 1);
            vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 1);
            vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1], 1);
            vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[1], 1);

            // Level 5:
            // x_re: 0->3, 32->35
            // y_re: 16->19, 48->51
            // x_im: 256->259, 288->291
            // y_im: 272->275, 304->307

            ////////
            // x - y
            ////////
            vfsubx4_swap(v_re, x_re, y_re, 0, 2, 1, 3);
            vfsubx4_swap(v_im, x_im, y_im, 0, 2, 1, 3);

            ////////
            // x + y
            ////////
            // x_re: 0->3, 16->19
            // x_im: 256->259, 272-> 275
            vfaddx4_swap(x_re, x_re, y_re, 0, 2, 1, 3);
            vfaddx4_swap(x_im, x_im, y_im, 0, 2, 1, 3);

            // Calculate y
            // v_im*s_im
            vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[2], 1);
            vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[2], 1);
            vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[2], 1);
            vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[2], 1);

            // v_im*s_im + v_re*s_re
            // y_re: 32->35, 48->51
            vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[2], 0);
            vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2], 0);
            vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[2], 0);
            vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2], 0);

            // v_im*s_re
            vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[2], 0);
            vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[2], 0);
            vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[2], 0);
            vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[2], 0);

            // v_im*s_re - v_re*s_im
            // y_im: 288->291, 304->307
            vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[2], 1);
            vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2], 1);
            vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[2], 1);
            vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2], 1);

            // Store
            x_tmp.val[0] = x_re.val[0];
            x_tmp.val[1] = x_re.val[1];
            vstorex2(&f[j], x_tmp);
            x_tmp.val[0] = x_re.val[2];
            x_tmp.val[1] = x_re.val[3];
            vstorex2(&f[j + 16], x_tmp);

            y_tmp.val[0] = y_re.val[0];
            y_tmp.val[1] = y_re.val[1];
            vstorex2(&f[j + 32], y_tmp);
            y_tmp.val[0] = y_re.val[2];
            y_tmp.val[1] = y_re.val[3];
            vstorex2(&f[j + 48], y_tmp);

            x_tmp.val[0] = x_im.val[0];
            x_tmp.val[1] = x_im.val[1];
            vstorex2(&f[j + hn], x_tmp);
            x_tmp.val[0] = x_im.val[2];
            x_tmp.val[1] = x_im.val[3];
            vstorex2(&f[j + hn + 16], x_tmp);

            y_tmp.val[0] = y_im.val[0];
            y_tmp.val[1] = y_im.val[1];
            vstorex2(&f[j + hn + 32], y_tmp);
            y_tmp.val[0] = y_im.val[2];
            y_tmp.val[1] = y_im.val[3];
            vstorex2(&f[j + hn + 48], y_tmp);
        }
    }

    vloadx2(s_tmp, &fpr_gm_tab[(FALCON_N) >> 6]);
    s_re_im.val[0] = s_tmp.val[0];
    s_re_im.val[1] = s_tmp.val[1];

    vload(s_re_im.val[2], &fpr_gm_tab[(FALCON_N) >> 7]);

    div_n = vdupq_n_f64(fpr_p2_tab[FALCON_LOGN]);
    vfmul(s_re_im.val[2], s_re_im.val[2], div_n);

    for (int j = 0; j < 64; j += 4)
    {
        // Level 6:
        vloadx2(x_tmp, &f[j]);
        x_re.val[0] = x_tmp.val[0];
        x_re.val[1] = x_tmp.val[1];
        vloadx2(y_tmp, &f[j + 64]);
        y_re.val[0] = y_tmp.val[0];
        y_re.val[1] = y_tmp.val[1];

        vloadx2(x_tmp, &f[j + 128]);
        x_re.val[2] = x_tmp.val[0];
        x_re.val[3] = x_tmp.val[1];
        vloadx2(y_tmp, &f[j + 192]);
        y_re.val[2] = y_tmp.val[0];
        y_re.val[3] = y_tmp.val[1];

        vloadx2(x_tmp, &f[j + hn]);
        x_im.val[0] = x_tmp.val[0];
        x_im.val[1] = x_tmp.val[1];
        vloadx2(y_tmp, &f[j + hn + 64]);
        y_im.val[0] = y_tmp.val[0];
        y_im.val[1] = y_tmp.val[1];

        vloadx2(x_tmp, &f[j + hn + 128]);
        x_im.val[2] = x_tmp.val[0];
        x_im.val[3] = x_tmp.val[1];
        vloadx2(y_tmp, &f[j + hn + 192]);
        y_im.val[2] = y_tmp.val[0];
        y_im.val[3] = y_tmp.val[1];

        ////////
        // x - y
        ////////
        vfsubx4(v_re, x_re, y_re);
        vfsubx4(v_im, x_im, y_im);

        ////////
        // x + y
        ////////
        vfaddx4(x_re, x_re, y_re);
        vfaddx4(x_im, x_im, y_im);

        // Calculate y
        // v_im*s_im
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 1);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 1);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[1], 1);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[1], 1);

        // v_im*s_im + v_re*s_re
        vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 0);
        vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 0);
        vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1], 0);
        vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[1], 0);

        // v_im*s_re
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 0);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 0);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[1], 0);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[1], 0);

        // v_im*s_re - v_re*s_im
        vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 1);
        vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 1);
        vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1], 1);
        vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[1], 1);

        // Level 7:
        ////////
        // x - y
        ////////
        vfsubx4_swap(v_re, x_re, y_re, 0, 2, 1, 3);
        vfsubx4_swap(v_im, x_im, y_im, 0, 2, 1, 3);

        ////////
        // x + y
        ////////
        vfaddx4_swap(x_re, x_re, y_re, 0, 2, 1, 3);
        vfaddx4_swap(x_im, x_im, y_im, 0, 2, 1, 3);

        // Calculate y
        // v_im*s_im
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[2], 1);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[2], 1);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[2], 1);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[2], 1);

        // v_im*s_im + v_re*s_re
        vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[2], 0);
        vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2], 0);
        vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[2], 0);
        vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2], 0);

        // v_im*s_re
        vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[2], 0);
        vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[2], 0);
        vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[2], 0);
        vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[2], 0);

        // v_im*s_re - v_re*s_im
        vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[2], 1);
        vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2], 1);
        vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[2], 1);
        vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2], 1);

        // Divide by N at the end
        vfmul(x_re.val[0], x_re.val[0], div_n);
        vfmul(x_re.val[1], x_re.val[1], div_n);
        vfmul(x_re.val[2], x_re.val[2], div_n);
        vfmul(x_re.val[3], x_re.val[3], div_n);

        vfmul(x_im.val[0], x_im.val[0], div_n);
        vfmul(x_im.val[1], x_im.val[1], div_n);
        vfmul(x_im.val[2], x_im.val[2], div_n);
        vfmul(x_im.val[3], x_im.val[3], div_n);

        // Store
        x_tmp.val[0] = x_re.val[0];
        x_tmp.val[1] = x_re.val[1];
        vstorex2(&f[j], x_tmp);
        x_tmp.val[0] = x_re.val[2];
        x_tmp.val[1] = x_re.val[3];
        vstorex2(&f[j + 64], x_tmp);

        y_tmp.val[0] = y_re.val[0];
        y_tmp.val[1] = y_re.val[1];
        vstorex2(&f[j + 128], y_tmp);
        y_tmp.val[0] = y_re.val[2];
        y_tmp.val[1] = y_re.val[3];
        vstorex2(&f[j + 192], y_tmp);

        x_tmp.val[0] = x_im.val[0];
        x_tmp.val[1] = x_im.val[1];
        vstorex2(&f[j + hn], x_tmp);
        x_tmp.val[0] = x_im.val[2];
        x_tmp.val[1] = x_im.val[3];
        vstorex2(&f[j + hn + 64], x_tmp);

        y_tmp.val[0] = y_im.val[0];
        y_tmp.val[1] = y_im.val[1];
        vstorex2(&f[j + hn + 128], y_tmp);
        y_tmp.val[0] = y_im.val[2];
        y_tmp.val[1] = y_im.val[3];
        vstorex2(&f[j + hn + 192], y_tmp);
    }
    // End function

    // Optional, combine two level 4-5, 6-7 loop, but the compiler generate overhead loop, so I discard
    /* 
    int distance;
    const unsigned int LAST_L = 6; // Last level
    for (int l = 4; l < 8; l += 2)
    {
        distance = 1 << l;

        for (int i = 0; i < FALCON_N/2; i += 1 << (l+2) )
        {
            vloadx2(s_tmp, &fpr_gm_tab[(FALCON_N + i) >> l]);
            s_re_im.val[0] = s_tmp.val[0];
            s_re_im.val[1] = s_tmp.val[1];
            vload(s_re_im.val[2], &fpr_gm_tab[(FALCON_N + i) >> (l+1)]);

            if (l == LAST_L)
            {
                vfmul(s_re_im.val[2], s_re_im.val[2], div_n);
            }
            
            for (int j = i; j < i + distance; j += 4)
            {
                vloadx2(x_tmp, &f[j]);
                x_re.val[0] = x_tmp.val[0];
                x_re.val[1] = x_tmp.val[1];
                vloadx2(y_tmp, &f[j + distance]);
                y_re.val[0] = y_tmp.val[0];
                y_re.val[1] = y_tmp.val[1];

                vloadx2(x_tmp, &f[j + 2*distance]);
                x_re.val[2] = x_tmp.val[0];
                x_re.val[3] = x_tmp.val[1];
                vloadx2(y_tmp, &f[j + 3*distance]);
                y_re.val[2] = y_tmp.val[0];
                y_re.val[3] = y_tmp.val[1];

                vloadx2(x_tmp, &f[j + hn]);
                x_im.val[0] = x_tmp.val[0];
                x_im.val[1] = x_tmp.val[1];
                vloadx2(y_tmp, &f[j + hn + distance]);
                y_im.val[0] = y_tmp.val[0];
                y_im.val[1] = y_tmp.val[1];

                vloadx2(x_tmp, &f[j + hn + 2*distance]);
                x_im.val[2] = x_tmp.val[0];
                x_im.val[3] = x_tmp.val[1];
                vloadx2(y_tmp, &f[j + hn + 3*distance]);
                y_im.val[2] = y_tmp.val[0];
                y_im.val[3] = y_tmp.val[1];

                ////////
                // x - y
                ////////
                vfsubx4(v_re, x_re, y_re);
                vfsubx4(v_im, x_im, y_im);

                ////////
                // x + y
                ////////
                // x_re: 0->3, 32->35
                // x_im: 256->259, 288->291
                vfaddx4(x_re, x_re, y_re);
                vfaddx4(x_im, x_im, y_im);

                // Calculate y
                // v_im*s_im
                vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 1);
                vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 1);
                vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[1], 1);
                vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[1], 1);

                // v_im*s_im + v_re*s_re
                // y_re: 16->19, 48->51
                vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 0);
                vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 0);
                vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1], 0);
                vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[1], 0);

                // v_im*s_re
                vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[0], 0);
                vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[0], 0);
                vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[1], 0);
                vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[1], 0);

                // v_im*s_re - v_re*s_im
                // y_im: 272->275, 304->307
                vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[0], 1);
                vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[0], 1);
                vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[1], 1);
                vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[1], 1);

                // Layer 6:
                // x_re: 0->3, 32->35
                // y_re: 16->19, 48->51
                // x_im: 256->259, 288->291
                // y_im: 272->275, 304->307

                ////////
                // x - y
                ////////
                vfsubx4_swap(v_re, x_re, y_re, 0, 2, 1, 3);
                vfsubx4_swap(v_im, x_im, y_im, 0, 2, 1, 3);

                ////////
                // x + y
                ////////
                // x_re: 0->3, 16->19
                // x_im: 256->259, 272-> 275
                vfaddx4_swap(x_re, x_re, y_re, 0, 2, 1, 3);
                vfaddx4_swap(x_im, x_im, y_im, 0, 2, 1, 3);

                // Calculate y
                // v_im*s_im
                vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[2], 1);
                vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[2], 1);
                vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[2], 1);
                vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[2], 1);

                // v_im*s_im + v_re*s_re
                // y_re: 32->35, 48->51
                vfma_lane(y_re.val[0], tmp.val[0], v_re.val[0], s_re_im.val[2], 0);
                vfma_lane(y_re.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2], 0);
                vfma_lane(y_re.val[2], tmp.val[2], v_re.val[2], s_re_im.val[2], 0);
                vfma_lane(y_re.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2], 0);

                // v_im*s_re
                vfmul_lane(tmp.val[0], v_im.val[0], s_re_im.val[2], 0);
                vfmul_lane(tmp.val[1], v_im.val[1], s_re_im.val[2], 0);
                vfmul_lane(tmp.val[2], v_im.val[2], s_re_im.val[2], 0);
                vfmul_lane(tmp.val[3], v_im.val[3], s_re_im.val[2], 0);

                // v_im*s_re - v_re*s_im
                // y_im: 288->291, 304->307
                vfms_lane(y_im.val[0], tmp.val[0], v_re.val[0], s_re_im.val[2], 1);
                vfms_lane(y_im.val[1], tmp.val[1], v_re.val[1], s_re_im.val[2], 1);
                vfms_lane(y_im.val[2], tmp.val[2], v_re.val[2], s_re_im.val[2], 1);
                vfms_lane(y_im.val[3], tmp.val[3], v_re.val[3], s_re_im.val[2], 1);

                if (l == LAST_L)
                {
                    vfmul(x_re.val[0], x_re.val[0], div_n);
                    vfmul(x_re.val[1], x_re.val[1], div_n);
                    vfmul(x_re.val[2], x_re.val[2], div_n);
                    vfmul(x_re.val[3], x_re.val[3], div_n);

                    vfmul(x_im.val[0], x_im.val[0], div_n);
                    vfmul(x_im.val[1], x_im.val[1], div_n);
                    vfmul(x_im.val[2], x_im.val[2], div_n);
                    vfmul(x_im.val[3], x_im.val[3], div_n);
                }

                // Store
                x_tmp.val[0] = x_re.val[0];
                x_tmp.val[1] = x_re.val[1];
                vstorex2(&f[j], x_tmp);
                x_tmp.val[0] = x_re.val[2];
                x_tmp.val[1] = x_re.val[3];
                vstorex2(&f[j + distance], x_tmp);

                y_tmp.val[0] = y_re.val[0];
                y_tmp.val[1] = y_re.val[1];
                vstorex2(&f[j + 2*distance], y_tmp);
                y_tmp.val[0] = y_re.val[2];
                y_tmp.val[1] = y_re.val[3];
                vstorex2(&f[j + 3*distance], y_tmp);

                x_tmp.val[0] = x_im.val[0];
                x_tmp.val[1] = x_im.val[1];
                vstorex2(&f[j + hn], x_tmp);
                x_tmp.val[0] = x_im.val[2];
                x_tmp.val[1] = x_im.val[3];
                vstorex2(&f[j + hn + distance], x_tmp);

                y_tmp.val[0] = y_im.val[0];
                y_tmp.val[1] = y_im.val[1];
                vstorex2(&f[j + hn + 2*distance], y_tmp);
                y_tmp.val[0] = y_im.val[2];
                y_tmp.val[1] = y_im.val[3];
                vstorex2(&f[j + hn + 3*distance], y_tmp);
            }
        }

    } */
}

void PQCLEAN_FALCON512_NEON_FFT(fpr *f)
{
    // Total: 32 = 16 + 8 + 8 register
    float64x2x4_t s_re_im, tmp;               // 8
    float64x2x4_t x_re, x_im, y_re, y_im;     // 16
    float64x2x4_t x1_re, x1_im, y1_re, y1_im; // 16
    float64x2x2_t s_tmp;                      // 2
    // Level 4, 5, 6, 7
    float64x2x2_t x_tmp, y_tmp;

    const unsigned int hn = FALCON_N >> 1;
    for (int l = 8; l > 4; l -= 2)
    {
        int distance = 1 << (l - 2);
        for (int i = 0; i < FALCON_N/2; i+= 1 << l)
        {
            vload(s_re_im.val[0], &fpr_gm_tab[(FALCON_N + i) >> (l - 1)]);
            vloadx2(s_tmp, &fpr_gm_tab[(FALCON_N + i) >> (l - 2)]);
            s_re_im.val[1] = s_tmp.val[0];
            s_re_im.val[2] = s_tmp.val[1];

            for (int j = i; j < i + distance; j += 4)
            {
                // Level 7
                // x1_re: 0->3, 64->67
                // x1_im: 256->259, 320 -> 323
                // y_re: 128->131, 192->195
                // y_im: 384->387, 448 -> 451
                vloadx2(x_tmp, &f[j]);
                x1_re.val[0] = x_tmp.val[0];
                x1_re.val[1] = x_tmp.val[1];
                vloadx2(x_tmp, &f[j + distance]);
                x1_re.val[2] = x_tmp.val[0];
                x1_re.val[3] = x_tmp.val[1];

                vloadx2(y_tmp, &f[j + 2*distance]);
                y_re.val[0] = y_tmp.val[0];
                y_re.val[1] = y_tmp.val[1];
                vloadx2(y_tmp, &f[j + 3*distance]);
                y_re.val[2] = y_tmp.val[0];
                y_re.val[3] = y_tmp.val[1];

                vloadx2(x_tmp, &f[j + hn]);
                x1_im.val[0] = x_tmp.val[0];
                x1_im.val[1] = x_tmp.val[1];
                vloadx2(x_tmp, &f[j + hn + distance]);
                x1_im.val[2] = x_tmp.val[0];
                x1_im.val[3] = x_tmp.val[1];

                vloadx2(y_tmp, &f[j + hn + 2*distance]);
                y_im.val[0] = y_tmp.val[0];
                y_im.val[1] = y_tmp.val[1];
                vloadx2(y_tmp, &f[j + hn + 3*distance]);
                y_im.val[2] = y_tmp.val[0];
                y_im.val[3] = y_tmp.val[1];

                vfmul_lane(y1_re.val[0], y_re.val[0], s_re_im.val[0], 0);
                vfmul_lane(y1_re.val[1], y_re.val[1], s_re_im.val[0], 0);
                vfmul_lane(y1_re.val[2], y_re.val[2], s_re_im.val[0], 0);
                vfmul_lane(y1_re.val[3], y_re.val[3], s_re_im.val[0], 0);

                vfms_lane(y1_re.val[0], y1_re.val[0],  y_im.val[0], s_re_im.val[0], 1);
                vfms_lane(y1_re.val[1], y1_re.val[1],  y_im.val[1], s_re_im.val[0], 1);
                vfms_lane(y1_re.val[2], y1_re.val[2],  y_im.val[2], s_re_im.val[0], 1);
                vfms_lane(y1_re.val[3], y1_re.val[3],  y_im.val[3], s_re_im.val[0], 1);

                vfmul_lane(y1_im.val[0], y_re.val[0], s_re_im.val[0], 1);
                vfmul_lane(y1_im.val[1], y_re.val[1], s_re_im.val[0], 1);
                vfmul_lane(y1_im.val[2], y_re.val[2], s_re_im.val[0], 1);
                vfmul_lane(y1_im.val[3], y_re.val[3], s_re_im.val[0], 1);

                vfma_lane(y1_im.val[0], y1_im.val[0],  y_im.val[0], s_re_im.val[0], 0);
                vfma_lane(y1_im.val[1], y1_im.val[1],  y_im.val[1], s_re_im.val[0], 0);
                vfma_lane(y1_im.val[2], y1_im.val[2],  y_im.val[2], s_re_im.val[0], 0);
                vfma_lane(y1_im.val[3], y1_im.val[3],  y_im.val[3], s_re_im.val[0], 0);

                vfaddx4(x_re, x1_re, y1_re);
                vfaddx4(x_im, x1_im, y1_im);

                vfsubx4(y_re, x1_re, y1_re);
                vfsubx4(y_im, x1_im, y1_im);
                
                // Level 6
                // x_re: 0->3, 64->67
                // y_re: 128->131, 192 -> 195
                // x_im: 256->259, 320 -> 323
                // y_im: 384->387, 448 -> 451
                
                vfmul_lane(y1_re.val[0], x_re.val[2], s_re_im.val[1], 0);
                vfmul_lane(y1_re.val[1], x_re.val[3], s_re_im.val[1], 0);
                vfmul_lane(y1_re.val[2], y_re.val[2], s_re_im.val[2], 0);
                vfmul_lane(y1_re.val[3], y_re.val[3], s_re_im.val[2], 0);

                vfms_lane(y1_re.val[0], y1_re.val[0], x_im.val[2], s_re_im.val[1], 1);
                vfms_lane(y1_re.val[1], y1_re.val[1], x_im.val[3], s_re_im.val[1], 1);
                vfms_lane(y1_re.val[2], y1_re.val[2], y_im.val[2], s_re_im.val[2], 1);
                vfms_lane(y1_re.val[3], y1_re.val[3], y_im.val[3], s_re_im.val[2], 1);

                vfmul_lane(y1_im.val[0], x_re.val[2], s_re_im.val[1], 1);
                vfmul_lane(y1_im.val[1], x_re.val[3], s_re_im.val[1], 1);
                vfmul_lane(y1_im.val[2], y_re.val[2], s_re_im.val[2], 1);
                vfmul_lane(y1_im.val[3], y_re.val[3], s_re_im.val[2], 1);

                vfma_lane(y1_im.val[0], y1_im.val[0],  x_im.val[2], s_re_im.val[1], 0);
                vfma_lane(y1_im.val[1], y1_im.val[1],  x_im.val[3], s_re_im.val[1], 0);
                vfma_lane(y1_im.val[2], y1_im.val[2],  y_im.val[2], s_re_im.val[2], 0);
                vfma_lane(y1_im.val[3], y1_im.val[3],  y_im.val[3], s_re_im.val[2], 0);

                vfadd(x1_re.val[0], x_re.val[0], y1_re.val[0]);
                vfadd(x1_re.val[1], x_re.val[1], y1_re.val[1]);
                vfadd(x1_re.val[2], y_re.val[0], y1_re.val[2]);
                vfadd(x1_re.val[3], y_re.val[1], y1_re.val[3]);

                vfadd(x1_im.val[0], x_im.val[0], y1_im.val[0]);
                vfadd(x1_im.val[1], x_im.val[1], y1_im.val[1]);
                vfadd(x1_im.val[2], y_im.val[0], y1_im.val[2]);
                vfadd(x1_im.val[3], y_im.val[1], y1_im.val[3]);

                vfsub(y1_re.val[0], x_re.val[0], y1_re.val[0]);
                vfsub(y1_re.val[1], x_re.val[1], y1_re.val[1]);
                vfsub(y1_re.val[2], y_re.val[0], y1_re.val[2]);
                vfsub(y1_re.val[3], y_re.val[1], y1_re.val[3]);

                vfsub(y1_im.val[0], x_im.val[0], y1_im.val[0]);
                vfsub(y1_im.val[1], x_im.val[1], y1_im.val[1]);
                vfsub(y1_im.val[2], y_im.val[0], y1_im.val[2]);
                vfsub(y1_im.val[3], y_im.val[1], y1_im.val[3]);
                
                // Level 6
                // x1_re: 0->3, 128 -> 131
                // y1_re: 64->67, 192 -> 195
                // x1_im: 256 -> 259, 384->387
                // y1_im: 320->323, 448 -> 451

                // Store
                x_tmp.val[0] = x1_re.val[0];
                x_tmp.val[1] = x1_re.val[1];
                vstorex2(&f[j], x_tmp);
                y_tmp.val[0] = y1_re.val[0];
                y_tmp.val[1] = y1_re.val[1];
                vstorex2(&f[j + distance], y_tmp);

                x_tmp.val[0] = x1_re.val[2];
                x_tmp.val[1] = x1_re.val[3];
                vstorex2(&f[j + 2*distance], x_tmp);
                y_tmp.val[0] = y1_re.val[2];
                y_tmp.val[1] = y1_re.val[3];
                vstorex2(&f[j + 3*distance], y_tmp);

                x_tmp.val[0] = x1_im.val[0];
                x_tmp.val[1] = x1_im.val[1];
                vstorex2(&f[j + hn], x_tmp);
                y_tmp.val[0] = y1_im.val[0];
                y_tmp.val[1] = y1_im.val[1];
                vstorex2(&f[j + hn + distance], y_tmp);

                x_tmp.val[0] = x1_im.val[2];
                x_tmp.val[1] = x1_im.val[3];
                vstorex2(&f[j + hn + 2*distance], x_tmp);
                y_tmp.val[0] = y1_im.val[2];
                y_tmp.val[1] = y1_im.val[3];
                vstorex2(&f[j + hn + 3*distance], y_tmp);
            }
        }
    }
    // End level 7, 6, 5, 4 loop
/* 
    vload(s_re_im.val[0], &fpr_gm_tab[FALCON_N >> 7]);
    vloadx2(s_tmp, &fpr_gm_tab[(FALCON_N) >> 6]);
    s_re_im.val[1] = s_tmp.val[0];
    s_re_im.val[2] = s_tmp.val[1];

    // Level 7, 6
    for (int j = 0; j < 64; j += 4)
    {
        // Level 7
        // x1_re: 0->3, 64->67
        // x1_im: 256->259, 320 -> 323
        // y_re: 128->131, 192->195
        // y_im: 384->387, 448 -> 451
        vloadx2(x_tmp, &f[j]);
        x1_re.val[0] = x_tmp.val[0];
        x1_re.val[1] = x_tmp.val[1];
        vloadx2(x_tmp, &f[j + 64]);
        x1_re.val[2] = x_tmp.val[0];
        x1_re.val[3] = x_tmp.val[1];

        vloadx2(y_tmp, &f[j + 128]);
        y_re.val[0] = y_tmp.val[0];
        y_re.val[1] = y_tmp.val[1];
        vloadx2(y_tmp, &f[j + 192]);
        y_re.val[2] = y_tmp.val[0];
        y_re.val[3] = y_tmp.val[1];

        vloadx2(x_tmp, &f[j + hn]);
        x1_im.val[0] = x_tmp.val[0];
        x1_im.val[1] = x_tmp.val[1];
        vloadx2(x_tmp, &f[j + hn + 64]);
        x1_im.val[2] = x_tmp.val[0];
        x1_im.val[3] = x_tmp.val[1];

        vloadx2(y_tmp, &f[j + hn + 128]);
        y_im.val[0] = y_tmp.val[0];
        y_im.val[1] = y_tmp.val[1];
        vloadx2(y_tmp, &f[j + hn + 192]);
        y_im.val[2] = y_tmp.val[0];
        y_im.val[3] = y_tmp.val[1];

        vfmul_lane(y1_re.val[0], y_re.val[0], s_re_im.val[0], 0);
        vfmul_lane(y1_re.val[1], y_re.val[1], s_re_im.val[0], 0);
        vfmul_lane(y1_re.val[2], y_re.val[2], s_re_im.val[0], 0);
        vfmul_lane(y1_re.val[3], y_re.val[3], s_re_im.val[0], 0);

        vfms_lane(y1_re.val[0], y1_re.val[0],  y_im.val[0], s_re_im.val[0], 1);
        vfms_lane(y1_re.val[1], y1_re.val[1],  y_im.val[1], s_re_im.val[0], 1);
        vfms_lane(y1_re.val[2], y1_re.val[2],  y_im.val[2], s_re_im.val[0], 1);
        vfms_lane(y1_re.val[3], y1_re.val[3],  y_im.val[3], s_re_im.val[0], 1);

        vfmul_lane(y1_im.val[0], y_re.val[0], s_re_im.val[0], 1);
        vfmul_lane(y1_im.val[1], y_re.val[1], s_re_im.val[0], 1);
        vfmul_lane(y1_im.val[2], y_re.val[2], s_re_im.val[0], 1);
        vfmul_lane(y1_im.val[3], y_re.val[3], s_re_im.val[0], 1);

        vfma_lane(y1_im.val[0], y1_im.val[0],  y_im.val[0], s_re_im.val[0], 0);
        vfma_lane(y1_im.val[1], y1_im.val[1],  y_im.val[1], s_re_im.val[0], 0);
        vfma_lane(y1_im.val[2], y1_im.val[2],  y_im.val[2], s_re_im.val[0], 0);
        vfma_lane(y1_im.val[3], y1_im.val[3],  y_im.val[3], s_re_im.val[0], 0);

        vfaddx4(x_re, x1_re, y1_re);
        vfaddx4(x_im, x1_im, y1_im);

        vfsubx4(y_re, x1_re, y1_re);
        vfsubx4(y_im, x1_im, y1_im);
        
        // Level 6
        // x_re: 0->3, 64->67
        // y_re: 128->131, 192 -> 195
        // x_im: 256->259, 320 -> 323
        // y_im: 384->387, 448 -> 451
        
        vfmul_lane(y1_re.val[0], x_re.val[2], s_re_im.val[1], 0);
        vfmul_lane(y1_re.val[1], x_re.val[3], s_re_im.val[1], 0);
        vfmul_lane(y1_re.val[2], y_re.val[2], s_re_im.val[2], 0);
        vfmul_lane(y1_re.val[3], y_re.val[3], s_re_im.val[2], 0);

        vfms_lane(y1_re.val[0], y1_re.val[0], x_im.val[2], s_re_im.val[1], 1);
        vfms_lane(y1_re.val[1], y1_re.val[1], x_im.val[3], s_re_im.val[1], 1);
        vfms_lane(y1_re.val[2], y1_re.val[2], y_im.val[2], s_re_im.val[2], 1);
        vfms_lane(y1_re.val[3], y1_re.val[3], y_im.val[3], s_re_im.val[2], 1);

        vfmul_lane(y1_im.val[0], x_re.val[2], s_re_im.val[1], 1);
        vfmul_lane(y1_im.val[1], x_re.val[3], s_re_im.val[1], 1);
        vfmul_lane(y1_im.val[2], y_re.val[2], s_re_im.val[2], 1);
        vfmul_lane(y1_im.val[3], y_re.val[3], s_re_im.val[2], 1);

        vfma_lane(y1_im.val[0], y1_im.val[0],  x_im.val[2], s_re_im.val[1], 0);
        vfma_lane(y1_im.val[1], y1_im.val[1],  x_im.val[3], s_re_im.val[1], 0);
        vfma_lane(y1_im.val[2], y1_im.val[2],  y_im.val[2], s_re_im.val[2], 0);
        vfma_lane(y1_im.val[3], y1_im.val[3],  y_im.val[3], s_re_im.val[2], 0);

        vfadd(x1_re.val[0], x_re.val[0], y1_re.val[0]);
        vfadd(x1_re.val[1], x_re.val[1], y1_re.val[1]);
        vfadd(x1_re.val[2], y_re.val[0], y1_re.val[2]);
        vfadd(x1_re.val[3], y_re.val[1], y1_re.val[3]);

        vfadd(x1_im.val[0], x_im.val[0], y1_im.val[0]);
        vfadd(x1_im.val[1], x_im.val[1], y1_im.val[1]);
        vfadd(x1_im.val[2], y_im.val[0], y1_im.val[2]);
        vfadd(x1_im.val[3], y_im.val[1], y1_im.val[3]);

        vfsub(y1_re.val[0], x_re.val[0], y1_re.val[0]);
        vfsub(y1_re.val[1], x_re.val[1], y1_re.val[1]);
        vfsub(y1_re.val[2], y_re.val[0], y1_re.val[2]);
        vfsub(y1_re.val[3], y_re.val[1], y1_re.val[3]);

        vfsub(y1_im.val[0], x_im.val[0], y1_im.val[0]);
        vfsub(y1_im.val[1], x_im.val[1], y1_im.val[1]);
        vfsub(y1_im.val[2], y_im.val[0], y1_im.val[2]);
        vfsub(y1_im.val[3], y_im.val[1], y1_im.val[3]);
        
        // Level 6
        // x1_re: 0->3, 128 -> 131
        // y1_re: 64->67, 192 -> 195
        // x1_im: 256 -> 259, 384->387
        // y1_im: 320->323, 448 -> 451

        // Store
        x_tmp.val[0] = x1_re.val[0];
        x_tmp.val[1] = x1_re.val[1];
        vstorex2(&f[j], x_tmp);
        y_tmp.val[0] = y1_re.val[0];
        y_tmp.val[1] = y1_re.val[1];
        vstorex2(&f[j + 64], y_tmp);

        x_tmp.val[0] = x1_re.val[2];
        x_tmp.val[1] = x1_re.val[3];
        vstorex2(&f[j + 128], x_tmp);
        y_tmp.val[0] = y1_re.val[2];
        y_tmp.val[1] = y1_re.val[3];
        vstorex2(&f[j + 192], y_tmp);

        x_tmp.val[0] = x1_im.val[0];
        x_tmp.val[1] = x1_im.val[1];
        vstorex2(&f[j + hn], x_tmp);
        y_tmp.val[0] = y1_im.val[0];
        y_tmp.val[1] = y1_im.val[1];
        vstorex2(&f[j + hn + 64], y_tmp);

        x_tmp.val[0] = x1_im.val[2];
        x_tmp.val[1] = x1_im.val[3];
        vstorex2(&f[j + hn + 128], x_tmp);
        y_tmp.val[0] = y1_im.val[2];
        y_tmp.val[1] = y1_im.val[3];
        vstorex2(&f[j + hn + 192], y_tmp);
    }


    // Level 5, 4
    for (int i = 0; i < FALCON_N/2; i+= 1<< 6)
    {
        vload(s_re_im.val[0], &fpr_gm_tab[(FALCON_N + i) >> 5]);
        vloadx2(s_tmp, &fpr_gm_tab[(FALCON_N + i) >> 4]);
        s_re_im.val[1] = s_tmp.val[0];
        s_re_im.val[2] = s_tmp.val[1];

        for (int j = i; j < i + 16; j += 4)
        {
            // Level 5
            // x1_re: 0->3, 64->67
            // x1_im: 256->259, 320 -> 323
            // y_re: 128->131, 192->195
            // y_im: 384->387, 448 -> 451
            vloadx2(x_tmp, &f[j]);
            x1_re.val[0] = x_tmp.val[0];
            x1_re.val[1] = x_tmp.val[1];
            vloadx2(x_tmp, &f[j + 16]);
            x1_re.val[2] = x_tmp.val[0];
            x1_re.val[3] = x_tmp.val[1];

            vloadx2(y_tmp, &f[j + 32]);
            y_re.val[0] = y_tmp.val[0];
            y_re.val[1] = y_tmp.val[1];
            vloadx2(y_tmp, &f[j + 48]);
            y_re.val[2] = y_tmp.val[0];
            y_re.val[3] = y_tmp.val[1];

            vloadx2(x_tmp, &f[j + hn]);
            x1_im.val[0] = x_tmp.val[0];
            x1_im.val[1] = x_tmp.val[1];
            vloadx2(x_tmp, &f[j + hn + 16]);
            x1_im.val[2] = x_tmp.val[0];
            x1_im.val[3] = x_tmp.val[1];

            vloadx2(y_tmp, &f[j + hn + 32]);
            y_im.val[0] = y_tmp.val[0];
            y_im.val[1] = y_tmp.val[1];
            vloadx2(y_tmp, &f[j + hn + 48]);
            y_im.val[2] = y_tmp.val[0];
            y_im.val[3] = y_tmp.val[1];

            vfmul_lane(y1_re.val[0], y_re.val[0], s_re_im.val[0], 0);
            vfmul_lane(y1_re.val[1], y_re.val[1], s_re_im.val[0], 0);
            vfmul_lane(y1_re.val[2], y_re.val[2], s_re_im.val[0], 0);
            vfmul_lane(y1_re.val[3], y_re.val[3], s_re_im.val[0], 0);

            vfms_lane(y1_re.val[0], y1_re.val[0],  y_im.val[0], s_re_im.val[0], 1);
            vfms_lane(y1_re.val[1], y1_re.val[1],  y_im.val[1], s_re_im.val[0], 1);
            vfms_lane(y1_re.val[2], y1_re.val[2],  y_im.val[2], s_re_im.val[0], 1);
            vfms_lane(y1_re.val[3], y1_re.val[3],  y_im.val[3], s_re_im.val[0], 1);

            vfmul_lane(y1_im.val[0], y_re.val[0], s_re_im.val[0], 1);
            vfmul_lane(y1_im.val[1], y_re.val[1], s_re_im.val[0], 1);
            vfmul_lane(y1_im.val[2], y_re.val[2], s_re_im.val[0], 1);
            vfmul_lane(y1_im.val[3], y_re.val[3], s_re_im.val[0], 1);

            vfma_lane(y1_im.val[0], y1_im.val[0],  y_im.val[0], s_re_im.val[0], 0);
            vfma_lane(y1_im.val[1], y1_im.val[1],  y_im.val[1], s_re_im.val[0], 0);
            vfma_lane(y1_im.val[2], y1_im.val[2],  y_im.val[2], s_re_im.val[0], 0);
            vfma_lane(y1_im.val[3], y1_im.val[3],  y_im.val[3], s_re_im.val[0], 0);

            vfaddx4(x_re, x1_re, y1_re);
            vfaddx4(x_im, x1_im, y1_im);

            vfsubx4(y_re, x1_re, y1_re);
            vfsubx4(y_im, x1_im, y1_im);
            
            // Level 6
            // x_re: 0->3, 64->67
            // y_re: 128->131, 192 -> 195
            // x_im: 256->259, 320 -> 323
            // y_im: 384->387, 448 -> 451
            
            vfmul_lane(y1_re.val[0], x_re.val[2], s_re_im.val[1], 0);
            vfmul_lane(y1_re.val[1], x_re.val[3], s_re_im.val[1], 0);
            vfmul_lane(y1_re.val[2], y_re.val[2], s_re_im.val[2], 0);
            vfmul_lane(y1_re.val[3], y_re.val[3], s_re_im.val[2], 0);

            vfms_lane(y1_re.val[0], y1_re.val[0], x_im.val[2], s_re_im.val[1], 1);
            vfms_lane(y1_re.val[1], y1_re.val[1], x_im.val[3], s_re_im.val[1], 1);
            vfms_lane(y1_re.val[2], y1_re.val[2], y_im.val[2], s_re_im.val[2], 1);
            vfms_lane(y1_re.val[3], y1_re.val[3], y_im.val[3], s_re_im.val[2], 1);

            vfmul_lane(y1_im.val[0], x_re.val[2], s_re_im.val[1], 1);
            vfmul_lane(y1_im.val[1], x_re.val[3], s_re_im.val[1], 1);
            vfmul_lane(y1_im.val[2], y_re.val[2], s_re_im.val[2], 1);
            vfmul_lane(y1_im.val[3], y_re.val[3], s_re_im.val[2], 1);

            vfma_lane(y1_im.val[0], y1_im.val[0],  x_im.val[2], s_re_im.val[1], 0);
            vfma_lane(y1_im.val[1], y1_im.val[1],  x_im.val[3], s_re_im.val[1], 0);
            vfma_lane(y1_im.val[2], y1_im.val[2],  y_im.val[2], s_re_im.val[2], 0);
            vfma_lane(y1_im.val[3], y1_im.val[3],  y_im.val[3], s_re_im.val[2], 0);

            vfadd(x1_re.val[0], x_re.val[0], y1_re.val[0]);
            vfadd(x1_re.val[1], x_re.val[1], y1_re.val[1]);
            vfadd(x1_re.val[2], y_re.val[0], y1_re.val[2]);
            vfadd(x1_re.val[3], y_re.val[1], y1_re.val[3]);

            vfadd(x1_im.val[0], x_im.val[0], y1_im.val[0]);
            vfadd(x1_im.val[1], x_im.val[1], y1_im.val[1]);
            vfadd(x1_im.val[2], y_im.val[0], y1_im.val[2]);
            vfadd(x1_im.val[3], y_im.val[1], y1_im.val[3]);

            vfsub(y1_re.val[0], x_re.val[0], y1_re.val[0]);
            vfsub(y1_re.val[1], x_re.val[1], y1_re.val[1]);
            vfsub(y1_re.val[2], y_re.val[0], y1_re.val[2]);
            vfsub(y1_re.val[3], y_re.val[1], y1_re.val[3]);

            vfsub(y1_im.val[0], x_im.val[0], y1_im.val[0]);
            vfsub(y1_im.val[1], x_im.val[1], y1_im.val[1]);
            vfsub(y1_im.val[2], y_im.val[0], y1_im.val[2]);
            vfsub(y1_im.val[3], y_im.val[1], y1_im.val[3]);
            
            // Level 6
            // x1_re: 0->3, 128 -> 131
            // y1_re: 64->67, 192 -> 195
            // x1_im: 256 -> 259, 384->387
            // y1_im: 320->323, 448 -> 451

            // Store
            x_tmp.val[0] = x1_re.val[0];
            x_tmp.val[1] = x1_re.val[1];
            vstorex2(&f[j], x_tmp);
            y_tmp.val[0] = y1_re.val[0];
            y_tmp.val[1] = y1_re.val[1];
            vstorex2(&f[j + 16], y_tmp);

            x_tmp.val[0] = x1_re.val[2];
            x_tmp.val[1] = x1_re.val[3];
            vstorex2(&f[j + 32], x_tmp);
            y_tmp.val[0] = y1_re.val[2];
            y_tmp.val[1] = y1_re.val[3];
            vstorex2(&f[j + 48], y_tmp);

            x_tmp.val[0] = x1_im.val[0];
            x_tmp.val[1] = x1_im.val[1];
            vstorex2(&f[j + hn], x_tmp);
            y_tmp.val[0] = y1_im.val[0];
            y_tmp.val[1] = y1_im.val[1];
            vstorex2(&f[j + hn + 16], y_tmp);

            x_tmp.val[0] = x1_im.val[2];
            x_tmp.val[1] = x1_im.val[3];
            vstorex2(&f[j + hn + 32], x_tmp);
            y_tmp.val[0] = y1_im.val[2];
            y_tmp.val[1] = y1_im.val[3];
            vstorex2(&f[j + hn + 48], y_tmp);
        }
    }
 */
    // End function
}