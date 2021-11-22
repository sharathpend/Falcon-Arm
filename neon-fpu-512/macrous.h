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
#define vload_u16_x4(c, addr) c = vld4q_u16(addr);

/* 
 * GS Buttefly with Barrett 1 unknown factor
 */
#define gsbf(a, b, zl, zh, N, t) \
    t = vsubq_u16(a, b);         \
    a = vaddq_u16(a, b);         \
    b = vmulq_u16(t, zh);        \
    t = vqrdmulhq_s16(t, zl);    \
    b = vmls_u16(t, N);

/* 
 * CT Buttefly with Barrett 1 unknown factor
 */
#define ctbf(a, b, zl, zh, N, t) \
    t = vmulq_s16(b, zh);        \
    b = vqrdmulhq_s16(b, zl);    \
    t = vmlsq_s16(b, N);         \
    b = vsubq_s16(a, t);         \
    a = vaddq_s16(a, t);

/* 
 * Montgomery multiplication via doubling with bNinv precomputed
 * t is temp register
 * Input: a, b, bNinv, N 
 * Output: c = ab * R^-1
 */
#define montmuld(c, a, b, bNinv, N, t) \
    c = vqdmulhq_s16(a, b);            \
    t = vmulq_s16(a, bNinv);           \
    t = vqdmulhq_s16(t, N);            \
    c = vhsubq_u16(c, t);

/* 
 * Montgomery multiplication via doubling without bNinv precomputed
 * t is temp register
 * Input: a, b, Ninv, N
 * Ouput: c = ab * R^-1
 */
#define montmuld_1(c, a, b, Ninv, N, t) \
    t = vmulq_s16(b, Ninv);             \
    c = vqdmulhq_s16(a, b);             \
    t = vmulq_s16(a, t);                \
    t = vqdmulhq_s16(t, N);             \
    c = vhsubq_u16(c, t);

/* 
 * Montgomery multiplication via rounding with -bNinv
 * t is temp register
 * Input: a, b, bNinv_neg (-bNinv), N
 * Output: c = 2ab * R^-1
 */
#define montmulr(c, a, b, bNinv_neg, N, t) \
    c = vqrdmulhq_s16(a, b);               \
    t = vmulq_s16(a, bNinv_neg);           \
    c = vqrdmlahq_s16(t, N);

/* 
 * Montgomery multiplication via rounding without -bNinv
 * t is temp register
 * Input: a, b, Ninv_neg (-Ninv), N
 * Output: c = 2ab * R^-1
 */
#define montmulr_1(c, a, b, Ninv_neg, N, t) \
    c = vqrdmulhq_s16(a, b);                \
    t = vmulq_s16(b, Ninv_neg);             \
    t = vmulq_s16(a, t);                    \
    c = vqrdmlahq_s16(t, N);

/*
 * Matrix 4x4 transpose: v
 * Input: int16x8x4_t v, tmp
 * Output: int16x8x4_t v
 */
#define transpose(v, tmp)                                                         \
  tmp.val[0] = vtrn1q_s16(v.val[0], v.val[1]);                                    \
  tmp.val[1] = vtrn2q_s16(v.val[0], v.val[1]);                                    \
  tmp.val[2] = vtrn1q_s16(v.val[2], v.val[3]);                                    \
  tmp.val[3] = vtrn2q_s16(v.val[2], v.val[3]);                                    \
  v.val[0] = (int16x8_t)vtrn1q_s32((int32x4_t)tmp.val[0], (int32x4_t)tmp.val[2]); \
  v.val[2] = (int16x8_t)vtrn2q_s32((int32x4_t)tmp.val[0], (int32x4_t)tmp.val[2]); \
  v.val[1] = (int16x8_t)vtrn1q_s32((int32x4_t)tmp.val[1], (int32x4_t)tmp.val[3]); \
  v.val[3] = (int16x8_t)vtrn2q_s32((int32x4_t)tmp.val[1], (int32x4_t)tmp.val[3]);

#endif
