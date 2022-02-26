/*
 * Macro for sign/unsigned integer
 *
 * =============================================================================
 * Copyright (c) 2021 by Cryptographic Engineering Research Group (CERG)
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

#ifndef MACROUS_H
#define MACROUS_H

#include <arm_neon.h>

#define vmull_lo(c, a, b) c = vmull_s16(vget_low_s16(a), vget_low_s16(b));

#define vmull_hi(c, a, b) c = vmull_high_s16(a, b);

#define vmulla_lo(d, c, a, b) d = vmlal_s16(c, vget_low_s16(a), vget_low_s16(b));

#define vmulla_hi(d, c, a, b) d = vmlal_high_s16(c, a, b);

#define vadd(c, a, b) c = vaddq_u32(a, b);

#define vaddv(c, a) c = vaddvq_u32(a);

#define vor(c, a, b) c = vorrq_u32(a, b);

#define vdup32x4(c, constant)         \
    c.val[0] = vdupq_n_u32(constant); \
    c.val[1] = vdupq_n_u32(constant); \
    c.val[2] = vdupq_n_u32(constant); \
    c.val[3] = vdupq_n_u32(constant);

// Macro for NTT operation. Using signed 16-bit.
#define vload_s16_4(c, addr) c = vld4q_s16(addr);
#define vload_s16_x2(c, addr) c = vld1q_s16_x2(addr);
#define vload_s16_x4(c, addr) c = vld1q_s16_x4(addr);

#define vstore_s16_x4(addr, c) vst1q_s16_x4(addr, c);
#define vstore_s16_x2(addr, c) vst1q_s16_x2(addr, c);

/*
 * Strategy for NTT:
 * - Forward and Inverse NTT multiply with constant, use either Barrett or Montgomery *Rounding* arithmetic
 * - Pointwise multiplication must use Montgomery *Doubling* arithmetic
 *
 * Rounding because:
 * - Montgomery need one coefficient to be *odd*, it only works with precomputed coefficient
 * - Barrett reduction work.
 *
 * Doubling because
 * - Montgomery Doubling work with two unknown coefficient, no constaint at all
 */

// ------------ Forward NTT and Inverse NTT ------------
/*
 * GS Butterfly with Montgomery *Rounding* reduction
 * Input: a, b < R/2
 * Output: c in [-q, q], c = a * (2bR^-1)
 */
#define gsbf_mt(a, b, zl, zh, N, t) \
    t = vsubq_s16(a, b);            \
    a = vaddq_s16(a, b);            \
    b = vqrdmulhq_s16(t, zl);       \
    t = vmulq_s16(t, zh);           \
    b = vqrdmlahq_s16(b, t, N);

#define gsbf_mti(a, b, zl, zh, N, t, il, ih) \
    t = vsubq_s16(a, b);                     \
    a = vaddq_s16(a, b);                     \
    b = vqrdmulhq_laneq_s16(t, zl, il);      \
    t = vmulq_laneq_s16(a, zh, ih);          \
    b = vqrdmlahq_s16(b, t, N);

/*
 * CT Butterfly with Montgomery *Rounding* reduction
 * Input: a, b < R/2
 * Output: c in [-q, q], c = a * (2bR^-1)
 */
#define ctbf_mt(a, b, zl, zh, N, t) \
    t = vqrdmulhq_s16(b, zl);       \
    b = vmulq_s16(b, zh);           \
    t = vqrdmlahq_s16(t, b, N);     \
    b = vsubq_s16(a, t);            \
    a = vaddq_s16(a, t);

#define ctbf_mti(a, b, zl, zh, N, t, il, iH) \
    t = vqrdmulhq_laneq_s16(b, zl, il);      \
    b = vmulq_laneq_s16(b, zh, ih);          \
    t = vqrdmlahq_s16(t, b, N);              \
    b = vsubq_s16(a, t);                     \
    a = vaddq_s16(a, t);

// ------------ Pointwise Multiplication ------------
/*
 * Montgomery multiplication via *Doubling*
 * Input: a, b, bNinv, N
 * Output: c = ab * R^-1
 */
#define montmul(c, a, b, t, neon_q, neon_qinv) \
    c = vqdmulhq_s16(a, b);                    \
    t = vmulq_s16(b, neon_qinv);               \
    t = vmulq_s16(a, t);                       \
    t = vqdmulhq_s16(t, neon_q);               \
    c = vhsubq_u16(c, t);

// ------------ Barrett Reduction ------------
/* Barrett reduction, return [-Q/2, Q/2]
 * `v` = 5461, `n` = 11
 */
#define barrett(a, t, neon_v, neon_q) \
    t = vqdmulhq_s16(a, neon_v);      \
    t = vrshrq_n_s16(t, 11);          \
    a = vmlsq_s16(a, t, neon_q);

// ------------ Matrix Transpose ------------
/*
 * Matrix 4x4 transpose: v
 * Input: int16x8x4_t v, tmp
 * Output: int16x8x4_t v
 */
#define transpose(v, tmp)                                                           \
    tmp.val[0] = vtrn1q_s16(v.val[0], v.val[1]);                                    \
    tmp.val[1] = vtrn2q_s16(v.val[0], v.val[1]);                                    \
    tmp.val[2] = vtrn1q_s16(v.val[2], v.val[3]);                                    \
    tmp.val[3] = vtrn2q_s16(v.val[2], v.val[3]);                                    \
    v.val[0] = (int16x8_t)vtrn1q_s32((int32x4_t)tmp.val[0], (int32x4_t)tmp.val[2]); \
    v.val[2] = (int16x8_t)vtrn2q_s32((int32x4_t)tmp.val[0], (int32x4_t)tmp.val[2]); \
    v.val[1] = (int16x8_t)vtrn1q_s32((int32x4_t)tmp.val[1], (int32x4_t)tmp.val[3]); \
    v.val[3] = (int16x8_t)vtrn2q_s32((int32x4_t)tmp.val[1], (int32x4_t)tmp.val[3]);

#endif
