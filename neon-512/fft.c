#include "inner.h"

// c <= addr interleave
#define vload4(c, addr) c = vld4q_f64(addr);
// c <= addr x4
#define vloadx4(c, addr) c = vld1q_f64_x4(addr);
// c <= addr x2
#define vload2(c, addr) c = vld2q_f64(addr);
// addr <= c
#define vstorex4(addr, c) vst1q_f64_x4(addr, c);

#define transpose(a, b, t, ia, ib, it)            \
    t.val[it] = a.val[ia];                        \
    a.val[ia] = vzip1q_f64(t.val[it], b.val[ib]); \
    b.val[ib] = vzip2q_f64(t.val[it], b.val[ib]);

// c = a - b
#define vsubx4(c, a, b)                       \
    c.val[0] = vsubq_f64(a.val[0], b.val[0]); \
    c.val[1] = vsubq_f64(a.val[1], b.val[1]); \
    c.val[2] = vsubq_f64(a.val[2], b.val[2]); \
    c.val[3] = vsubq_f64(a.val[3], b.val[3]);

// c = a + b
#define vaddx4(c, a, b)                       \
    c.val[0] = vaddq_f64(a.val[0], b.val[0]); \
    c.val[1] = vaddq_f64(a.val[1], b.val[1]); \
    c.val[2] = vaddq_f64(a.val[2], b.val[2]); \
    c.val[3] = vaddq_f64(a.val[3], b.val[3]);

#define vsubx4_mix(c, a, b, ia0, ia1, ia2, ia3)   \
    c.val[0] = vsubq_f64(a.val[ia0], a.val[ia2]); \
    c.val[1] = vsubq_f64(a.val[ia1], a.val[ia3]); \
    c.val[2] = vsubq_f64(b.val[ia0], b.val[ia2]); \
    c.val[3] = vsubq_f64(b.val[ia1], b.val[ia3]);

#define vaddx4_mix(c, a, b, ia0, ia1, ia2, ia3)   \
    c.val[0] = vaddq_f64(a.val[ia0], a.val[ia2]); \
    c.val[1] = vaddq_f64(a.val[ia1], a.val[ia3]); \
    c.val[2] = vaddq_f64(b.val[ia0], b.val[ia2]); \
    c.val[3] = vaddq_f64(b.val[ia1], b.val[ia3]);

// c = a * b[i]
#define vfmul_lane(c, a, b, ib0, ib1, ib2, ib3, i0, i1, i2, i3) \
    c.val[0] = vmulq_laneq_f64(a.val[0], b.val[ib0], i0);       \
    c.val[1] = vmulq_laneq_f64(a.val[1], b.val[ib1], i1);       \
    c.val[2] = vmulq_laneq_f64(a.val[2], b.val[ib2], i2);       \
    c.val[3] = vmulq_laneq_f64(a.val[3], b.val[ib3], i3);

// d = c + a*b
#define vfma_lane(d, c, a, b, ib0, ib1, ib2, ib3, i0, i1, i2, i3)   \
    d.val[0] = vfmaq_laneq_f64(c.val[0], a.val[0], b.val[ib0], i0); \
    d.val[1] = vfmaq_laneq_f64(c.val[1], a.val[1], b.val[ib1], i1); \
    d.val[2] = vfmaq_laneq_f64(c.val[2], a.val[2], b.val[ib2], i2); \
    d.val[3] = vfmaq_laneq_f64(c.val[3], a.val[3], b.val[ib3], i3);

// d = c - a*b
#define vfms_lane(d, c, a, b, ib0, ib1, ib2, ib3, i0, i1, i2, i3)   \
    d.val[0] = vfmsq_laneq_f64(c.val[0], a.val[0], b.val[ib0], i0); \
    d.val[1] = vfmsq_laneq_f64(c.val[1], a.val[1], b.val[ib1], i1); \
    d.val[2] = vfmsq_laneq_f64(c.val[2], a.val[2], b.val[ib2], i2); \
    d.val[3] = vfmsq_laneq_f64(c.val[3], a.val[3], b.val[ib3], i3);

void PQCLEAN_FALCON512_NEON_iFFT(fpr *f, unsigned logn)
{
    // Total: 32 registers
    float64x2x4_t x_y_re[2], x_y_im[2], s_re_im[2];   // 24
    float64x2x4_t x_re, x_im, y_re, y_im; // 0
    float64x2x4_t v_re, v_im;             // 8
    float64x2x4_t tmp;                    // 4
    float64x2x2_t s_tmp;                  // 2

    const unsigned FALCON_N = 1 << logn;

    const unsigned hn = FALCON_N >> 1;
    unsigned m = FALCON_N, t = 1;
    unsigned hm = m >> 1, dt = t << 1;
    // Layer 1, 2, 3, 4
    for (int j = 0; j < FALCON_N/2; j += 16)
    {
        // Level 1
        // x_re = 0, 4 | 2, 6 | 8, 12 | 10, 14
        // y_re = 1, 5 | 3, 7 | 9, 13 | 11, 15
        // x_im = 256, 260 | 258, 262 | 264, 268 | 266, 270
        // y_im = 257, 261 | 259, 263 | 265, 269 | 267, 271
        vload4(x_y_re[0], &f[j]);
        vload4(x_y_re[1], &f[j + 8]);
        vload4(x_y_im[0], &f[j + hn]);
        vload4(x_y_im[1], &f[j + hn + 8]);
        // x_y_re[0] = vld4q_f64(&f[j]);
        // x_y_im[0] = vld4q_f64(&f[j + hn]);
        // x_y_re[1] = vld4q_f64(&f[j + 8]);
        // x_y_im[1] = vld4q_f64(&f[j + hn + 8]);

        // This assignment is free
        x_re.val[0] = x_y_re[0].val[0];
        x_re.val[1] = x_y_re[0].val[2];
        x_re.val[2] = x_y_re[1].val[0];
        x_re.val[3] = x_y_re[1].val[2];

        y_re.val[0] = x_y_re[0].val[1];
        y_re.val[1] = x_y_re[0].val[3];
        y_re.val[2] = x_y_re[1].val[1];
        y_re.val[3] = x_y_re[1].val[3];

        x_im.val[0] = x_y_im[0].val[0];
        x_im.val[1] = x_y_im[0].val[2];
        x_im.val[2] = x_y_im[1].val[0];
        x_im.val[3] = x_y_im[1].val[2];

        y_im.val[0] = x_y_im[0].val[1];
        y_im.val[1] = x_y_im[0].val[3];
        y_im.val[2] = x_y_im[1].val[1];
        y_im.val[3] = x_y_im[1].val[3];

        ////////
        // x - y
        ////////

        //  1,5 <= 0,4  -  1,5 |   3,7 <=   2,6 -  3,7
        // 9,13 <= 8,12 - 9,13 | 11,15 <= 10,14 - 11,15
        vsubx4(v_re, x_re, y_re);
        // v_re.val[0] = vsubq_f64(x_re.val[0], y_re.val[0]);
        // v_re.val[1] = vsubq_f64(x_re.val[1], y_re.val[1]);
        // v_re.val[2] = vsubq_f64(x_re.val[2], y_re.val[2]);
        // v_re.val[3] = vsubq_f64(x_re.val[3], y_re.val[3]);

        // 257,261 <= 256,260 - 257,261 | 259,263 <= 258,262 - 259,263
        // 265,269 <= 264,268 - 265,269 | 267,271 <= 266,270 - 267,271
        vsubx4(v_im, x_im, y_im);
        // v_im.val[0] = vsubq_f64(x_im.val[0], y_im.val[0]);
        // v_im.val[1] = vsubq_f64(x_im.val[1], y_im.val[1]);
        // v_im.val[2] = vsubq_f64(x_im.val[2], y_im.val[2]);
        // v_im.val[3] = vsubq_f64(x_im.val[3], y_im.val[3]);

        ////////
        // x + y
        ////////

        // x_re: 0,4 | 2,6 | 8,12 | 10,14
        // 0,4  <= 0,4  +  1,5 |   2,6 <=   2,6 +   3,7
        // 8,12 <= 8,12 + 9,13 | 10,14 <= 10,14 + 11,15
        vaddx4(x_re, x_re, y_re);
        // x_re.val[0] = vaddq_f64(x_re.val[0], y_re.val[0]);
        // x_re.val[1] = vaddq_f64(x_re.val[1], y_re.val[1]);
        // x_re.val[2] = vaddq_f64(x_re.val[2], y_re.val[2]);
        // x_re.val[3] = vaddq_f64(x_re.val[3], y_re.val[3]);

        // x_im: 256,260 | 258,262 | 264,268 | 266,270
        // 256,260 <= 256,260 + 257,261 | 258,262 <= 258,262 + 259,263
        // 264,268 <= 264,268 + 265,269 | 266,270 <= 266,270 + 267,271
        vaddx4(x_im, x_im, y_im);
        // x_im.val[0] = vaddq_f64(x_im.val[0], y_im.val[0]);
        // x_im.val[1] = vaddq_f64(x_im.val[1], y_im.val[1]);
        // x_im.val[2] = vaddq_f64(x_im.val[2], y_im.val[2]);
        // x_im.val[3] = vaddq_f64(x_im.val[3], y_im.val[3]);

        // s * (x - y) = s*v = (s_re + i*s_im)(v_re + i*v_im)
        // re:  v_re*s_re + v_im*s_im
        // im: -v_re*s_im + v_im*s_re

        // s_re_im = 512 -> 519
        // s_re: 512 -> 518, step 2
        // s_im: 513 -> 519, step 2
        vload4(s_re_im[0], &fpr_gm_tab[FALCON_N + j]);
        vload4(s_re_im[1], &fpr_gm_tab[FALCON_N + j + 8]);

        // Calculate y
        // v_im*s_im
        
        tmp.val[0] = vmulq_f64(v_im.val[0], s_re_im[0].val[1]);
        tmp.val[1] = vmulq_f64(v_im.val[1], s_re_im[0].val[3]);
        tmp.val[2] = vmulq_f64(v_im.val[2], s_re_im[1].val[1]);
        tmp.val[3] = vmulq_f64(v_im.val[3], s_re_im[1].val[3]);

        // v_im*s_im + v_re*s_re
        // y_re: 1,5 | 3,7 | 9,13 | 11,15
        y_re.val[0] = vfmaq_f64(tmp.val[0], v_re.val[0], s_re_im[0].val[0]);
        y_re.val[1] = vfmaq_f64(tmp.val[1], v_re.val[1], s_re_im[0].val[2]);
        y_re.val[2] = vfmaq_f64(tmp.val[2], v_re.val[2], s_re_im[1].val[0]);
        y_re.val[3] = vfmaq_f64(tmp.val[3], v_re.val[3], s_re_im[1].val[2]);

        // v_im*s_re
        tmp.val[0] = vmulq_f64(v_im.val[0], s_re_im[0].val[0]);
        tmp.val[1] = vmulq_f64(v_im.val[1], s_re_im[0].val[2]);
        tmp.val[2] = vmulq_f64(v_im.val[2], s_re_im[1].val[0]);
        tmp.val[3] = vmulq_f64(v_im.val[3], s_re_im[1].val[2]);

        // v_im*s_re - v_re*s_im
        // y_im: 257,261 | 259,263 | 265,269 | 267,271
        y_im.val[0] = vfmsq_f64(tmp.val[0], v_re.val[0], s_re_im[0].val[1]);
        y_im.val[1] = vfmsq_f64(tmp.val[1], v_re.val[1], s_re_im[0].val[3]);
        y_im.val[2] = vfmsq_f64(tmp.val[2], v_re.val[2], s_re_im[1].val[1]);
        y_im.val[3] = vfmsq_f64(tmp.val[3], v_re.val[3], s_re_im[1].val[3]);

        // print_vector(x_re);
        // print_vector(y_re);


        ////////////////////// Level 2
        // x_re = 0,4 | 2,6 | 8,12 | 10,14
        // y_re = 1,5 | 3,7 | 9,13 | 11,15
        // x_im = 256,260 | 258,262 | 264,268 | 266,270
        // y_im = 257,261 | 259,263 | 265,269 | 267,271

        vload2(s_tmp, &fpr_gm_tab[(FALCON_N >> 1) + j]);
        s_re_im[0].val[0] = s_tmp.val[0];
        s_re_im[0].val[1] = s_tmp.val[1];
        vload2(s_tmp, &fpr_gm_tab[(FALCON_N >> 1) + j + 4]);
        s_re_im[0].val[2] = s_tmp.val[0];
        s_re_im[0].val[3] = s_tmp.val[1];
        ////////
        // x - y
        ////////

        // 0,4 - 2,6 | 8,12 - 10,14
        // 1,5 - 3,7 | 9,13 - 11,15

        v_re.val[0] = vsubq_f64(x_re.val[0], x_re.val[1]);
        v_re.val[1] = vsubq_f64(x_re.val[2], x_re.val[3]);
        v_re.val[2] = vsubq_f64(y_re.val[0], y_re.val[1]);
        v_re.val[3] = vsubq_f64(y_re.val[2], y_re.val[3]);

        // 256,260 - 258,262 | 264,268 - 266,270
        // 257,261 - 259,263 | 265,269 - 267,271
        v_im.val[0] = vsubq_f64(x_im.val[0], x_im.val[1]);
        v_im.val[1] = vsubq_f64(x_im.val[2], x_im.val[3]);
        v_im.val[2] = vsubq_f64(y_im.val[0], y_im.val[1]);
        v_im.val[3] = vsubq_f64(y_im.val[2], y_im.val[3]);

        ////////
        // x + y
        ////////

        // x_re: 0,4 | 8,12 | 1,5 | 9,13
        // 0,4 <= 0,4 + 2,6 | 8,12 <= 8,12 + 10,14
        // 1,5 <= 1,5 + 3,7 | 9,13 <= 9,13 + 11,15
        x_re.val[0] = vaddq_f64(x_re.val[0], x_re.val[1]);
        x_re.val[1] = vaddq_f64(x_re.val[2], x_re.val[3]);
        x_re.val[2] = vaddq_f64(y_re.val[0], y_re.val[1]);
        x_re.val[3] = vaddq_f64(y_re.val[2], y_re.val[3]);

        // x_im: 256, 260 | 264, 268 | 257, 261 | 265, 269
        // 256,260 <= 256,260 + 258,262 | 264,268 <= 264,268 + 266,270
        // 257,261 <= 257,261 + 259,263 | 265,269 <= 265,269 + 267,271
        x_im.val[0] = vaddq_f64(x_im.val[0], x_im.val[1]);
        x_im.val[1] = vaddq_f64(x_im.val[2], x_im.val[3]);
        x_im.val[2] = vaddq_f64(y_im.val[0], y_im.val[1]);
        x_im.val[3] = vaddq_f64(y_im.val[2], y_im.val[3]);

        // Calculate y
        // v_im*s_im
        tmp.val[0] = vmulq_f64(v_im.val[0], s_re_im[0].val[1]);
        tmp.val[1] = vmulq_f64(v_im.val[1], s_re_im[0].val[3]);
        tmp.val[2] = vmulq_f64(v_im.val[2], s_re_im[0].val[1]);
        tmp.val[3] = vmulq_f64(v_im.val[3], s_re_im[0].val[3]);

        // v_im*s_im + v_re*s_re
        // y_re: 2,6 | 10,14 | 3,7 | 11,15
        y_re.val[0] = vfmaq_f64(tmp.val[0], v_re.val[0], s_re_im[0].val[0]);
        y_re.val[1] = vfmaq_f64(tmp.val[1], v_re.val[1], s_re_im[0].val[2]);
        y_re.val[2] = vfmaq_f64(tmp.val[2], v_re.val[2], s_re_im[0].val[0]);
        y_re.val[3] = vfmaq_f64(tmp.val[3], v_re.val[3], s_re_im[0].val[2]);

        // v_im*s_re
        tmp.val[0] = vmulq_f64(v_im.val[0], s_re_im[0].val[0]);
        tmp.val[1] = vmulq_f64(v_im.val[1], s_re_im[0].val[2]);
        tmp.val[2] = vmulq_f64(v_im.val[2], s_re_im[0].val[0]);
        tmp.val[3] = vmulq_f64(v_im.val[3], s_re_im[0].val[2]);

        // v_im*s_re - v_re*s_im
        // y_im: 258,262 | 266,270 | 259,263 | 267,271
        y_im.val[0] = vfmsq_f64(tmp.val[0], v_re.val[0], s_re_im[0].val[1]);
        y_im.val[1] = vfmsq_f64(tmp.val[1], v_re.val[1], s_re_im[0].val[3]); 
        y_im.val[2] = vfmsq_f64(tmp.val[2], v_re.val[2], s_re_im[0].val[1]);
        y_im.val[3] = vfmsq_f64(tmp.val[3], v_re.val[3], s_re_im[0].val[3]);

        // Level 3
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
        
        vload2(s_tmp, &fpr_gm_tab[(FALCON_N >> 2) + j]);

        ////////
        // x - y
        ////////
        // TODO: DEBUG HERE

        // 0,1 - 4,5   | 2,3 - 6,7
        // 8,9 - 12,13 | 10,11 - 14,15
        v_re.val[0] = vsubq_f64(x_re.val[0], x_re.val[2]);
        v_re.val[2] = vsubq_f64(x_re.val[1], x_re.val[3]); 
        v_re.val[1] = vsubq_f64(y_re.val[0], y_re.val[2]);
        v_re.val[3] = vsubq_f64(y_re.val[1], y_re.val[3]);

        // 256,257 - 260,261 | 258,259 - 262,263
        // 264,265 - 268,269 | 266,267 - 270,271
        v_im.val[0] = vsubq_f64(x_im.val[0], x_im.val[2]);
        v_im.val[2] = vsubq_f64(x_im.val[1], x_im.val[3]);
        v_im.val[1] = vsubq_f64(y_im.val[0], y_im.val[2]);
        v_im.val[3] = vsubq_f64(y_im.val[1], y_im.val[3]);

        ////////
        // x + y
        ////////

        // x_re: 0,1 | 2,3 |  8,9 | 10,11
        // 0,1 <= 0,1 + 4,5  | 2,3 <= 2,3 + 6,7 
        // 8,9 <= 8,9 + 12,13| 10,11 <= 10,11 + 14,15
        x_re.val[0] = vaddq_f64(x_re.val[0], x_re.val[2]);
        x_re.val[2] = vaddq_f64(x_re.val[1], x_re.val[3]);
        x_re.val[1] = vaddq_f64(y_re.val[0], y_re.val[2]);
        x_re.val[3] = vaddq_f64(y_re.val[1], y_re.val[3]);

        // x_im: 256,257 | 258,259 | 264,265 | 266,267
        // 256,257 <= 256,257 + 260,261 | 258,259 <= 258,259 + 262,263
        // 264,265 <= 264,265 + 268,269 | 266,267 <= 266,267 + 270,271
        x_im.val[0] = vaddq_f64(x_im.val[0], x_im.val[2]);
        x_im.val[2] = vaddq_f64(x_im.val[1], x_im.val[3]);
        x_im.val[1] = vaddq_f64(y_im.val[0], y_im.val[2]);
        x_im.val[3] = vaddq_f64(y_im.val[1], y_im.val[3]);

        // Calculate y
        // v_im*s_im
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_tmp.val[1], 0);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_tmp.val[1], 0);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_tmp.val[1], 1);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_tmp.val[1], 1);

        // v_im*s_im + v_re*s_re
        // y_re: 4,5 | 6,7 | 12,13 | 14,15
        y_re.val[0] = vfmaq_laneq_f64(tmp.val[0], v_re.val[0], s_tmp.val[0], 0);
        y_re.val[1] = vfmaq_laneq_f64(tmp.val[1], v_re.val[1], s_tmp.val[0], 0);
        y_re.val[2] = vfmaq_laneq_f64(tmp.val[2], v_re.val[2], s_tmp.val[0], 1);
        y_re.val[3] = vfmaq_laneq_f64(tmp.val[3], v_re.val[3], s_tmp.val[0], 1);

        // v_im*s_re
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_tmp.val[0], 0);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_tmp.val[0], 0);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_tmp.val[0], 1);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_tmp.val[0], 1);

        // v_im*s_re - v_re*s_im
        // y_im: 260,261 |  262,263 | 268,269 | 270,271
        y_im.val[0] = vfmsq_laneq_f64(tmp.val[0], v_re.val[0], s_tmp.val[1], 0);
        y_im.val[1] = vfmsq_laneq_f64(tmp.val[1], v_re.val[1], s_tmp.val[1], 0);
        y_im.val[2] = vfmsq_laneq_f64(tmp.val[2], v_re.val[2], s_tmp.val[1], 1);
        y_im.val[3] = vfmsq_laneq_f64(tmp.val[3], v_re.val[3], s_tmp.val[1], 1);

        
        // Level 4
        // x_re: 0,1 | 2,3 | 8,9 | 10,11
        // y_re: 4,5 | 6,7 | 12,13 | 14,15
        // x_im: 256,257 | 258,259 | 264,265 | 266,267
        // y_im: 260,261 | 262,263 | 268,269 | 270,271

        // Load s_re_im
        s_tmp.val[0] = vld1q_f64(&fpr_gm_tab[(FALCON_N >> 3) + j]);
        ////////
        // x - y
        ////////

        // 0,1 -   8,9 | 2,3 - 10,11
        // 4,5 - 12,13 | 6,7 - 14,15
        v_re.val[0] = vsubq_f64(x_re.val[0], x_re.val[2]);
        v_re.val[1] = vsubq_f64(x_re.val[1], x_re.val[3]);
        v_re.val[2] = vsubq_f64(y_re.val[0], y_re.val[2]);
        v_re.val[3] = vsubq_f64(y_re.val[1], y_re.val[3]);

        // 256,257 - 264,265 | 258,259 - 266,267
        // 260,261 - 268,269 | 262,263 - 270,271
        v_im.val[0] = vsubq_f64(x_im.val[0], x_im.val[2]);
        v_im.val[1] = vsubq_f64(x_im.val[1], x_im.val[3]);
        v_im.val[2] = vsubq_f64(y_im.val[0], y_im.val[2]);
        v_im.val[3] = vsubq_f64(y_im.val[1], y_im.val[3]);

        ////////
        // x + y
        ////////

        // x_re: 0,1 | 2,3 | 4,5 | 6,7
        // 0,1 <= 0,1 +   8,9 | 2,3 <= 2,3 + 10,11
        // 4,5 <= 4,5 + 12,13 | 6,7 <= 6,7 + 14,15
        x_re.val[0] = vaddq_f64(x_re.val[0], x_re.val[2]);
        x_re.val[1] = vaddq_f64(x_re.val[1], x_re.val[3]);
        x_re.val[2] = vaddq_f64(y_re.val[0], y_re.val[2]);
        x_re.val[3] = vaddq_f64(y_re.val[1], y_re.val[3]);

        // x_im: 256,257 | 258,259 | 260,261 | 262,263
        // 256,257 <= 256,257 + 264,265 | 258,259 <= 258,259 + 266,267
        // 260,261 <= 260,261 + 268,269 | 262,263 <= 262,263 + 270,271
        x_im.val[0] = vaddq_f64(x_im.val[0], x_im.val[2]);
        x_im.val[1] = vaddq_f64(x_im.val[1], x_im.val[3]);
        x_im.val[2] = vaddq_f64(y_im.val[0], y_im.val[2]);
        x_im.val[3] = vaddq_f64(y_im.val[1], y_im.val[3]);

        // Calculate y
        // v_im*s_im
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_tmp.val[0], 1);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_tmp.val[0], 1);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_tmp.val[0], 1);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_tmp.val[0], 1);

        // v_im*s_im + v_re*s_re
        // y_re: 8,9 | 10,11 | 12,13 | 14,15
        y_re.val[0] = vfmaq_laneq_f64(tmp.val[0], v_re.val[0], s_tmp.val[0], 0);
        y_re.val[1] = vfmaq_laneq_f64(tmp.val[1], v_re.val[1], s_tmp.val[0], 0);
        y_re.val[2] = vfmaq_laneq_f64(tmp.val[2], v_re.val[2], s_tmp.val[0], 0);
        y_re.val[3] = vfmaq_laneq_f64(tmp.val[3], v_re.val[3], s_tmp.val[0], 0);

        // v_im*s_re
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_tmp.val[0], 0);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_tmp.val[0], 0);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_tmp.val[0], 0);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_tmp.val[0], 0);

        // v_im*s_re - v_re*s_im
        // y_im: 264,265 | 266,267 | 268,269 | 270,271
        y_im.val[0] = vfmsq_laneq_f64(tmp.val[0], v_re.val[0], s_tmp.val[0], 1);
        y_im.val[1] = vfmsq_laneq_f64(tmp.val[1], v_re.val[1], s_tmp.val[0], 1);
        y_im.val[2] = vfmsq_laneq_f64(tmp.val[2], v_re.val[2], s_tmp.val[0], 1);
        y_im.val[3] = vfmsq_laneq_f64(tmp.val[3], v_re.val[3], s_tmp.val[0], 1);

        // x_re: 0,1 | 2,3 | 4,5 | 6,7
        // y_re: 8,9 | 10,11 | 12,13 | 14,15
        // x_im: 256,257 | 258,259 | 260,261 | 262,263
        // y_im: 264,265 | 266,267 | 268,269 | 270,271
        vstorex4(&f[j], x_re);
        vstorex4(&f[j + 8], y_re);
        vstorex4(&f[j + hn], x_im);
        vstorex4(&f[j + hn + 8], y_im);
    }
    /* 
    // Layer 5, 6
    // TODO: update the bound fix bound
    float64x2x2_t l5_x_re, l5_y_re, l5_x_im, l5_y_im;
    for (int i = 0; i < 256; i+= 64)
    for (int j = i; j < i + 16; j += 4)
    {
        // Layer 5
        // x_re: 0 ->3, 32->35
        // y_re: 16 -> 19, 48 -> 51
        // x_im: 256 -> 259, 288-> 291
        // y_im: 272 -> 275, 304 -> 307
        l5_x_re = vld1q_f64_x2(&f[j]);
        x_re.val[0] = l5_x_re.val[0];
        x_re.val[1] = l5_x_re.val[1];
        l5_x_re = vld1q_f64_x2(&f[j + 32]);
        x_re.val[2] = l5_x_re.val[0];
        x_re.val[3] = l5_x_re.val[1];

        l5_y_re = vld1q_f64_x2(&f[j + 16]);
        y_re.val[0] = l5_y_re.val[0];
        y_re.val[1] = l5_y_re.val[1];
        l5_y_re = vld1q_f64_x2(&f[j + 48]);
        y_re.val[2] = l5_y_re.val[0];
        y_re.val[3] = l5_y_re.val[1];

        l5_x_im = vld1q_f64_x2(&f[j + hn]);
        x_im.val[0] = l5_x_im.val[0];
        x_im.val[1] = l5_x_im.val[1];
        l5_x_im = vld1q_f64_x2(&f[j + hn + 32]);
        x_im.val[2] = l5_x_im.val[0];
        x_im.val[3] = l5_x_im.val[1];

        l5_y_im = vld1q_f64_x2(&f[j + hn + 16]);
        y_im.val[0] = l5_y_im.val[0];
        y_im.val[1] = l5_y_im.val[1];
        l5_y_im = vld1q_f64_x2(&f[j + hn + 48]);
        y_im.val[2] = l5_y_im.val[0];
        y_im.val[3] = l5_y_im.val[1];

        // TODO: fix this
        // 32, 33, 34, 35
        l5_x_re = vld1q_f64_x2(&fpr_gm_tab[(FALCON_N >> 4)]);
        s_re_im.val[0] = l5_x_re.val[0];
        s_re_im.val[1] = l5_x_re.val[1];

        ////////
        // x - y
        ////////
        vsubx4(v_re, x_re, y_re);
        vsubx4(v_im, x_im, y_im);

        ////////
        // x + y
        ////////
        // x_re: 0->3, 32->35
        // x_im: 256->259, 288->291
        vaddx4(x_re, x_re, y_re);
        vaddx4(x_im, x_im, y_im);

        // Calculate y
        // v_im*s_im
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_re_im.val[0], 1);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_re_im.val[1], 1);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_re_im.val[2], 1);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_re_im.val[3], 1);

        // v_im*s_im + v_re*s_re
        // y_re: 16->19, 48->51
        y_re.val[0] = vfmaq_laneq_f64(tmp.val[0], v_re.val[0], s_re_im.val[0], 0);
        y_re.val[1] = vfmaq_laneq_f64(tmp.val[1], v_re.val[1], s_re_im.val[1], 0);
        y_re.val[2] = vfmaq_laneq_f64(tmp.val[2], v_re.val[2], s_re_im.val[2], 0);
        y_re.val[3] = vfmaq_laneq_f64(tmp.val[3], v_re.val[3], s_re_im.val[3], 0);

        // v_im*s_re
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_re_im.val[0], 0);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_re_im.val[1], 0);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_re_im.val[2], 0);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_re_im.val[3], 0);

        // v_im*s_re - v_re*s_im
        // y_im: 272->275, 304->307
        y_im.val[0] = vfmsq_laneq_f64(tmp.val[0], v_re.val[0], s_re_im.val[0], 1);
        y_im.val[1] = vfmsq_laneq_f64(tmp.val[1], v_re.val[1], s_re_im.val[1], 1);
        y_im.val[2] = vfmsq_laneq_f64(tmp.val[2], v_re.val[2], s_re_im.val[2], 1);
        y_im.val[3] = vfmsq_laneq_f64(tmp.val[3], v_re.val[3], s_re_im.val[3], 1);

        // Layer 6:
        // x_re: 0->3, 32->35
        // y_re: 16->19, 48->51
        // x_im: 256->259, 288->291
        // y_im: 272->275, 304->307

        ////////
        // x - y
        ////////
        v_re.val[0] = vsubq_f64(x_re.val[0], x_re.val[2]);
        v_re.val[1] = vsubq_f64(x_re.val[1], x_re.val[3]);
        v_re.val[2] = vsubq_f64(y_re.val[0], y_re.val[2]);
        v_re.val[3] = vsubq_f64(y_re.val[1], y_re.val[3]);

        v_im.val[0] = vsubq_f64(x_im.val[0], x_im.val[2]);
        v_im.val[1] = vsubq_f64(x_im.val[1], x_im.val[3]);
        v_im.val[2] = vsubq_f64(y_im.val[0], y_im.val[2]);
        v_im.val[3] = vsubq_f64(y_im.val[1], y_im.val[3]);

        ////////
        // x + y
        ////////
        // x_re: 0->3, 16->19
        // x_im: 256->259, 272-> 275
        x_re.val[0] = vaddq_f64(x_re.val[0], x_re.val[2]);
        x_re.val[1] = vaddq_f64(x_re.val[1], x_re.val[3]);
        x_re.val[2] = vaddq_f64(y_re.val[0], y_re.val[2]);
        x_re.val[3] = vaddq_f64(y_re.val[1], y_re.val[3]);

        x_im.val[0] = vaddq_f64(x_im.val[0], x_im.val[2]);
        x_im.val[1] = vaddq_f64(x_im.val[1], x_im.val[3]);
        x_im.val[2] = vaddq_f64(y_im.val[0], y_im.val[2]);
        x_im.val[3] = vaddq_f64(y_im.val[1], y_im.val[3]);

        // Calculate y
        // v_im*s_im
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_re_im.val[0], 1);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_re_im.val[1], 1);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_re_im.val[2], 1);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_re_im.val[3], 1);

        // v_im*s_im + v_re*s_re
        // y_re: 32->35, 48->51
        y_re.val[0] = vfmaq_laneq_f64(tmp.val[0], v_re.val[0], s_re_im.val[0], 0);
        y_re.val[1] = vfmaq_laneq_f64(tmp.val[1], v_re.val[1], s_re_im.val[1], 0);
        y_re.val[2] = vfmaq_laneq_f64(tmp.val[2], v_re.val[2], s_re_im.val[2], 0);
        y_re.val[3] = vfmaq_laneq_f64(tmp.val[3], v_re.val[3], s_re_im.val[3], 0);

        // v_im*s_re
        tmp.val[0] = vmulq_laneq_f64(v_im.val[0], s_re_im.val[0], 0);
        tmp.val[1] = vmulq_laneq_f64(v_im.val[1], s_re_im.val[1], 0);
        tmp.val[2] = vmulq_laneq_f64(v_im.val[2], s_re_im.val[2], 0);
        tmp.val[3] = vmulq_laneq_f64(v_im.val[3], s_re_im.val[3], 0);

        // v_im*s_re - v_re*s_im
        // y_im: 288->291, 304->307
        y_im.val[0] = vfmsq_laneq_f64(tmp.val[0], v_re.val[0], s_re_im.val[0], 1);
        y_im.val[1] = vfmsq_laneq_f64(tmp.val[1], v_re.val[1], s_re_im.val[1], 1);
        y_im.val[2] = vfmsq_laneq_f64(tmp.val[2], v_re.val[2], s_re_im.val[2], 1);
        y_im.val[3] = vfmsq_laneq_f64(tmp.val[3], v_re.val[3], s_re_im.val[3], 1);

        // Store
        l5_x_re.val[0] = x_re.val[0];
        l5_x_re.val[1] = x_re.val[1];
        vst1q_f64_x2(&f[j], l5_x_re);
        l5_x_re.val[0] = x_re.val[2];
        l5_x_re.val[1] = x_re.val[3];
        vst1q_f64_x2(&f[j + 16], l5_x_re);

        l5_y_re.val[0] = y_re.val[0];
        l5_y_re.val[1] = y_re.val[1];
        vst1q_f64_x2(&f[j + 32], l5_y_re);
        l5_y_re.val[0] = y_re.val[2];
        l5_y_re.val[1] = y_re.val[3];
        vst1q_f64_x2(&f[j + 48], l5_y_re);

        l5_x_im.val[0] = x_im.val[0];
        l5_x_im.val[1] = x_im.val[1];
        vst1q_f64_x2(&f[j + hn], l5_x_im);
        l5_x_im.val[0] = x_im.val[2];
        l5_x_im.val[1] = x_im.val[3];
        vst1q_f64_x2(&f[j + hn + 16], l5_x_im);

        l5_y_im.val[0] = y_im.val[0];
        l5_y_im.val[1] = y_im.val[1];
        vst1q_f64_x2(&f[j + hn + 32], l5_y_im);
        l5_y_im.val[0] = y_im.val[2];
        l5_y_im.val[1] = y_im.val[3];
        vst1q_f64_x2(&f[j + hn + 48], l5_x_im);
    }
 */

    // Layer 7, 8
    for (int i = 0; i < 256; i += 128)
        for (int j = i; j < i + 64; j += 4)
        {
        }

    // End function
}