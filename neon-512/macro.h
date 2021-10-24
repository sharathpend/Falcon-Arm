#ifndef PQCLEAN_FALCON512_NEON_MACRO_H
#define PQCLEAN_FALCON512_NEON_MACRO_H
#include <arm_neon.h>

// c <= addr x1
#define vload(c, addr) c = vld1q_f64(addr);
// c <= addr interleave 4
#define vload4(c, addr) c = vld4q_f64(addr);
// c <= addr x2
#define vloadx2(c, addr) c = vld1q_f64_x2(addr);
// c <= addr x3
#define vloadx3(c, addr) c = vld1q_f64_x3(addr);
// c <= addr x4
#define vloadx4(c, addr) c = vld1q_f64_x4(addr);
// c <= addr interleave 2
#define vload2(c, addr) c = vld2q_f64(addr);
// addr <= c
#define vstore2(addr, c) vst2q_f64(addr, c);
// addr <= c
#define vstorex2(addr, c) vst1q_f64_x2(addr, c);
// addr <= c
#define vstorex4(addr, c) vst1q_f64_x4(addr, c);
// addr <= c
#define vstore4(addr, c) vst4q_f64(addr, c);
// c = a - b
#define vfsub(c, a, b) c = vsubq_f64(a, b);
// c = a + b
#define vfadd(c, a, b) c = vaddq_f64(a, b);
// c = a * b
#define vfmul(c, a, b) c = vmulq_f64(a, b);
// c = a * n (n is constant)
#define vfmuln(c, a, n) c = vmulq_n_f64(a, n);
// Swap from a|b to b|a
#define vswap(c, a) c = vextq_f64(a, a, 1);
// c = a * n (n is constant)
#define vfmulnx4(c, a, n)                \
    c.val[0] = vmulq_n_f64(a.val[0], n); \
    c.val[1] = vmulq_n_f64(a.val[1], n); \
    c.val[2] = vmulq_n_f64(a.val[2], n); \
    c.val[3] = vmulq_n_f64(a.val[3], n);
// d = c + a *b
#define vfma(d, c, a, b) d = vfmaq_f64(c, a, b);
// d = c - a * b
#define vfms(d, c, a, b) d = vfmsq_f64(c, a, b);
// c = a * b[i]
#define vfmul_lane(c, a, b, i) c = vmulq_laneq_f64(a, b, i);
// d = c + a * b[i]
#define vfma_lane(d, c, a, b, i) d = vfmaq_laneq_f64(c, a, b, i);
// d = c - a * b[i]
#define vfms_lane(d, c, a, b, i) d = vfmsq_laneq_f64(c, a, b, i);
// c = -a
#define vfneg(c, a) c = vnegq_f64(a);

#define transpose(a, b, t, ia, ib, it)            \
    t.val[it] = a.val[ia];                        \
    a.val[ia] = vzip1q_f64(t.val[it], b.val[ib]); \
    b.val[ib] = vzip2q_f64(t.val[it], b.val[ib]);

// c = a - b
#define vfsubx4(c, a, b)                      \
    c.val[0] = vsubq_f64(a.val[0], b.val[0]); \
    c.val[1] = vsubq_f64(a.val[1], b.val[1]); \
    c.val[2] = vsubq_f64(a.val[2], b.val[2]); \
    c.val[3] = vsubq_f64(a.val[3], b.val[3]);

// c = a + b
#define vfaddx4(c, a, b)                      \
    c.val[0] = vaddq_f64(a.val[0], b.val[0]); \
    c.val[1] = vaddq_f64(a.val[1], b.val[1]); \
    c.val[2] = vaddq_f64(a.val[2], b.val[2]); \
    c.val[3] = vaddq_f64(a.val[3], b.val[3]);

#define vfsubx4_swap(c, a, b, i0, i1, i2, i3)   \
    c.val[0] = vsubq_f64(a.val[i0], a.val[i1]); \
    c.val[1] = vsubq_f64(a.val[i2], a.val[i3]); \
    c.val[2] = vsubq_f64(b.val[i0], b.val[i1]); \
    c.val[3] = vsubq_f64(b.val[i2], b.val[i3]);

#define vfaddx4_swap(c, a, b, i0, i1, i2, i3)   \
    c.val[0] = vaddq_f64(a.val[i0], a.val[i1]); \
    c.val[1] = vaddq_f64(a.val[i2], a.val[i3]); \
    c.val[2] = vaddq_f64(b.val[i0], b.val[i1]); \
    c.val[3] = vaddq_f64(b.val[i2], b.val[i3]);

// fpr_floor
#define vfpr_floor(out, x, tmp)                                       \
    tmp.val[0] = vcvtq_s64_f64(x);           /* [0] = (int64_t) x */  \
    tmp.val[1] = vcvtq_f64_s64(tmp.val[0]);  /* [1] = (double) [0] */ \
    tmp.val[1] = vcgtq_f64(tmp.val[1], x);   /* [1] = [1] > x */      \
    out = vsubq_s64(tmp.val[0], tmp.val[1]); /* [0] - [1] */

// convert float64x2_t to int64x2_t
#define vfpr_of(out, x) out = vcvtq_f64_s64(x);


#include <stdio.h>

// fpr_expm_p63
static inline uint64x2_t
// static inline float64x2_t
vfpr_expm_p63(float64x2_t x, float64x2_t ccs)
{
    float64x2_t y, z, y0, y1, x2ab, x2ba, ab2, b2a, a2b, x4ab;
    uint64x2_t ret;
    float64x2x4_t neon_c07; // C[0->7]
    float64x2x3_t neon_c8b; // C[8->11]

    static const double C_expm[] = {
        0.000000002073772366009083061987,  // c12
        -0.000000025299506379442070029551, // c11
        0.000000275607356160477811864927,  // c10
        -0.000002755586350219122514855659, // c9
        0.000024801566833585381209939524,  // c8
        -0.000198412739277311890541063977, // c7
        0.001388888894063186997887560103,  // c6
        -0.008333333327800835146903501993, // c5
        0.041666666666110491190622155955,  // c4
        -0.166666666666984014666397229121, // c3
        0.500000000000019206858326015208,  // c2
        -0.999999999999994892974086724280, // c1
        1.0, 
        0.0
        // 0.000000002073772366009083061987,  // c12
        // 0.000000025299506379442070029551, // c11
        // 0.000000275607356160477811864927,  // c10
        // 0.000002755586350219122514855659, // c9
        // 0.000024801566833585381209939524,  // c8
        // 0.000198412739277311890541063977, // c7
        // 0.001388888894063186997887560103,  // c6
        // 0.008333333327800835146903501993, // c5
        // 0.041666666666110491190622155955,  // c4
        // 0.166666666666984014666397229121, // c3
        // 0.500000000000019206858326015208,  // c2
        // 0.999999999999994892974086724280, // c1
        // -1.0,
        // 1.0,
        // 1.0,
        // -1.0
    };

    vloadx4(neon_c07, &C_expm[0]);
    vloadx3(neon_c8b, &C_expm[8]);

    // neon_c07:
    // 0: 12, 11
    // 1: 10, 9
    // 2:  8, 7
    // 3:  6, 5

    // neon_c8b
    // 0: 4, 3
    // 1: 2, 1

    // x^2 = x*x = (a^2) | (b^2)
    vfmul(x2ab, x, x);
    // Swap lane, a^2|b^2 to b^2|a^2
    vswap(x2ba, x2ab);
    printf("x = %.20f|%.20f\n", x[0], x[1]);
    a2b = vzip2q_f64(x2ba, x);
    ab2 = vzip1q_f64(x, x2ba);

    printf("a2b = %.20f|%.20f\n", a2b[0], a2b[1]);
    printf("ab2 = %.20f|%.20f\n", ab2[0], ab2[1]);
    // vfmul(b2a, b2a, neon_c8b.val[2]);
    // vfmul(a2b, a2b, neon_c8b.val[2]);
    /* 
	 * Horner's two fold:
	 * even: (c12x^2 + c10)x^2 + c8)x^2 + c6)x^2 + c4)x^2 + c2)x^2 + c0
	 * odd : (c11x^2 + c09)x^2 + c7)x^2 + c5)x^2 + c3)x^2 + c1)x
	 * First Loop:
	 * a_y0 = (c12a^2 + c10)a^2 + c8)a^2 + c6)a^2 + c4)a^2 + c2
	 * b_y0 = (c11b^2 + c09)b^2 + c7)b^2 + c5)b^2 + c3)b^2 + c1
	 * Second loop: Swap (a^2) | (b^2) to (b^2) | (a^2)
	 * b_y1 = (c12b^2 + c10)b^2 + c8)b^2 + c6)b^2 + c4)b^2 + c2
	 * a_y1 = (c11a^2 + c09)a^2 + c7)a^2 + c5)a^2 + c3)a^2 + c1
	 * 
     * Swap: (b_y1)|(a_y1) to (a_y1)|(b_y1)
     * a_y1 = (c11a^2 + c09)a^2 + c7)a^2 + c5)a^2 + c3)a^2 + c1)a
     * b_y1 = (c12b^2 + c10)b^2 + c8)b^2 + c6)b^2 + c4)b^2 + c2)b^2
     * 
     * a_y0 = (c12a^2 + c10)a^2 + c8)a^2 + c6)a^2 + c4)a^2 + c2)a^2
     * b_y0 = (c11b^2 + c09)b^2 + c7)b^2 + c5)b^2 + c3)b^2 + c1)b
	 * Result:
	 * a_y = a_y0*a2b + a_y1*ab2 + c0
	 * b_y = b_y0*a2b + b_y1*ab2 + c0
	 */

    printf("neon_c07[0] = %.20f|%.20f\n", neon_c07.val[0][0], neon_c07.val[0][1]);
    printf("neon_c07[1] = %.20f|%.20f\n", neon_c07.val[1][0], neon_c07.val[1][1]);
    printf("ab2 = %.20f|%.20f\n", ab2[0], ab2[1]);
    // Compiler auto re-arrange these instructions
    vfma(y0, neon_c07.val[1], x2ab, neon_c07.val[0]);
    printf("y0 = %.20f|%.20f\n", y0[0], y0[1]);
    vfma(y0, neon_c07.val[2], x2ab, y0);
    printf("y0 = %.20f|%.20f\n", y0[0], y0[1]);
    vfma(y0, neon_c07.val[3], x2ab, y0);
    printf("y0 = %.20f|%.20f\n", y0[0], y0[1]);
    vfma(y0, neon_c8b.val[0], x2ab, y0);
    printf("y0 = %.20f|%.20f\n", y0[0], y0[1]);
    vfma(y0, neon_c8b.val[1], x2ab, y0);
    printf("y0 = %.20f|%.20f\n", y0[0], y0[1]);

    vfma(y1, neon_c07.val[1], x2ba, neon_c07.val[0]);
    printf("y1 = %.20f|%.20f\n", y1[0], y1[1]);
    vfma(y1, neon_c07.val[2], x2ba, y1);
    printf("y1 = %.20f|%.20f\n", y1[0], y1[1]);
    vfma(y1, neon_c07.val[3], x2ba, y1);
    printf("y1 = %.20f|%.20f\n", y1[0], y1[1]);
    vfma(y1, neon_c8b.val[0], x2ba, y1);
    printf("y1 = %.20f|%.20f\n", y1[0], y1[1]);
    vfma(y1, neon_c8b.val[1], x2ba, y1);
    printf("y1 = %.20f|%.20f\n", y1[0], y1[1]);

    vswap(y1, y1);

    printf("y1 = %.20f|%.20f\n", y1[0], y1[1]);

    z = vdupq_n_f64(1.0);
    // printf("z = %.20f|%.20f\n", z[0], z[1]);
    vfma(y, z, y0, a2b);

    printf("y = %.20f|%.20f\n", y[0], y[1]);
    
    vfma(y, y, y1, ab2);

    printf("y = %.20f|%.20f\n", y[0], y[1]);

    // TODO: debug this
    z = vdupq_n_f64(fpr_ptwo63);
    vfmul(y, y, ccs);
    printf("ycss = %.20f|%.20f\n", y[0], y[1]);
    vfmul(y, y, z);
    printf("yz = %.20f|%.20f\n\n", y[0], y[1]);

    ret = vcvtq_u64_f64(y);
    return ret;
}

#endif
