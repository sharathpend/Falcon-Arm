#ifndef PQCLEAN_FALCON512_NEON_VFPR_H
#define PQCLEAN_FALCON512_NEON_VFPR_H

#include <arm_neon.h>
#include "inner.h"

typedef double fpr;
typedef float64x2_t fprx2;

static inline fprx2
vfpr_of(int64x2_t i)
{
	return vcvtq_f64_s64(i);
}

static const double fprx2_q              = 12289.0;
static const double fprx2_inverse_of_q   = 1.0 / 12289.0;
static const double fprx2_inv_2sqrsigma0 = .150865048875372721532312163019;
static const double fprx2_inv_sigma      = .005819826392951607426919370871;
static const double fprx2_sigma_min_9    = 1.291500756233514568549480827642;
static const double fprx2_sigma_min_10   = 1.311734375905083682667395805765;
static const double fprx2_log2           = 0.69314718055994530941723212146;
static const double fprx2_inv_log2       = 1.4426950408889634073599246810;
static const double fprx2_bnorm_max      = 16822.4121;
static const double fprx2_zero           = 0.0;
static const double fprx2_one            = 1.0;
static const double fprx2_two            = 2.0;
static const double fprx2_onehalf        = 0.5;
static const double fprx2_invsqrt2       = 0.707106781186547524400844362105;
static const double fprx2_invsqrt8       = 0.353553390593273762200422181052;
static const double fprx2_ptwo31         = 2147483648.0;
static const double fprx2_ptwo31m1       = 2147483647.0;
static const double fprx2_mtwo31m1       = -2147483647.0;
static const double fprx2_ptwo63m1       = 9223372036854775807.0;
static const double fprx2_mtwo63m1       = -9223372036854775807.0;
static const double fprx2_ptwo63         = 9223372036854775808.0;

static inline int64x2_t
vfpr_rint(fprx2 x)
{
    return vcvtnq_s64_f64(x);
}

static inline int64x2_t
vfpr_floor(fprx2 x)
{
    int64x2_t r;
    uint64x2_t u;
    fprx2 tmp;
    r = vcvtq_s64_f64(x); 
    tmp = vcvtq_f64_s64(r);
    u = vcltq_f64(x, tmp);

    return vaddq_s64(r, (int64x2_t)u);
}

static inline int64x2_t
vfpr_trunc(fprx2 x)
{
    return vcvtq_s64_f64(x);
}

static inline fprx2
vfpr_add(fprx2 x, fprx2 y)
{
    return vaddq_f64(x, y);
}

static inline fprx2
vfpr_sub(fprx2 x, fprx2 y)
{
    return vsubq_f64(x, y);
}

static inline fprx2
vfpr_neg(fprx2 x)
{
    return vnegq_f64(x);
}

static inline fprx2
vfpr_half(fprx2 x)
{
    return vmulq_n_f64(x, fprx2_onehalf);
}

static inline fprx2
vfpr_double(fprx2 x)
{
    return vaddq_f64(x, x);
}

static inline fprx2
vfpr_mul(fprx2 x, fprx2 y)
{
    return vmulq_f64(x, y);
}

static inline fprx2
vfpr_sqr(fprx2 x)
{
    return vmulq_f64(x, x);
}

static inline fprx2
vfpr_inv(fprx2 x)
{
    return vdivq_f64(vdupq_n_f64(fprx2_one), x);
}

static inline fprx2
vfpr_div(fprx2 x, fprx2 y)
{
    return vdivq_f64(x, y);
}

static inline fprx2
vfpr_sqrt(fprx2 x)
{
    return vsqrtq_f64(x);
}

static inline fprx2
vfpr_sqrte(fprx2 x)
{
    return vrsqrteq_f64(x);
}

static inline uint64x2_t
vfpr_lt(fprx2 x, fprx2 y)
{
    return vcltq_f64(x, y);
}

// fpr_expm_p63
static inline uint64x2_t
vfpr_expm_p63(fprx2 x, fprx2 ccs)
{
    fprx2 y, z;
    float64x2x4_t neon_exp[3];

    /* 
     * Horner's method 2-fold failed (add/sub separate), 3-fold failed (add/sub mix)
     * due to overflow
     * 
	 * Horner's two fold: FAILED
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

    static const float64_t C_expm[] = {
        0.000000002073772366009083061987, // c12
        0.000000002073772366009083061987, // c12
        0.000000025299506379442070029551, // c11
        0.000000025299506379442070029551, // c11
        0.000000275607356160477811864927, // c10
        0.000000275607356160477811864927, // c10
        0.000002755586350219122514855659, // c9
        0.000002755586350219122514855659, // c9
        0.000024801566833585381209939524, // c8
        0.000024801566833585381209939524, // c8
        0.000198412739277311890541063977, // c7
        0.000198412739277311890541063977, // c7
        0.001388888894063186997887560103, // c6
        0.001388888894063186997887560103, // c6
        0.008333333327800835146903501993, // c5
        0.008333333327800835146903501993, // c5
        0.041666666666110491190622155955, // c4
        0.041666666666110491190622155955, // c4
        0.166666666666984014666397229121, // c3
        0.166666666666984014666397229121, // c3
        0.500000000000019206858326015208, // c2
        0.500000000000019206858326015208, // c2
        0.999999999999994892974086724280, // c1
        0.999999999999994892974086724280, // c1
    };
    neon_exp[0] = vld1q_f64_x4(&C_expm[0]);
    neon_exp[1] = vld1q_f64_x4(&C_expm[8]);
    neon_exp[2] = vld1q_f64_x4(&C_expm[16]);

    y = vfmsq_f64(neon_exp[0].val[1], neon_exp[0].val[0], x);
    y = vfmsq_f64(neon_exp[0].val[2], y, x);
    y = vfmsq_f64(neon_exp[0].val[3], y, x);
    y = vfmsq_f64(neon_exp[1].val[0], y, x);
    y = vfmsq_f64(neon_exp[1].val[1], y, x);
    y = vfmsq_f64(neon_exp[1].val[2], y, x);
    y = vfmsq_f64(neon_exp[1].val[3], y, x);
    y = vfmsq_f64(neon_exp[2].val[0], y, x);
    y = vfmsq_f64(neon_exp[2].val[1], y, x);
    y = vfmsq_f64(neon_exp[2].val[2], y, x);
    y = vfmsq_f64(neon_exp[2].val[3], y, x);
    y = vfmsq_f64(vdupq_n_f64(fprx2_one), y, x);

    z = vmulq_n_f64(ccs, fprx2_ptwo63);
    y = vmulq_f64(y, z);

    return vcvtq_u64_f64(y);
}

extern const fpr fpr_gm_tab[];

extern const fpr fpr_p2_tab[];

// Precompute for splitFFT
extern const fpr fpr_gm_tab_half[];


#endif
