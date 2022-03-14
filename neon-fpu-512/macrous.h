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

// Macro for NTT operation. Using signed 16-bit.
#define vload_s16_4(c, addr) c = vld4q_s16(addr);
#define vload_s16_x2(c, addr) c = vld1q_s16_x2(addr);
#define vload_s16_x4(c, addr) c = vld1q_s16_x4(addr);

#define vstore_s16_x4(addr, c) vst1q_s16_x4(addr, c);
#define vstore_s16_x2(addr, c) vst1q_s16_x2(addr, c);
#define vstore_s16_4(add, c) vst4q_s16(add, c);

/*
 * Strategy for NTT:
 * - Forward and Inverse NTT multiply with constant, use either Barrett or Montgomery *Rounding* arithmetic
 * - Pointwise multiplication must use Montgomery *Doubling* arithmetic
 *
 * Rounding because:
 *
 * - Montgomery need one coefficient to be *odd*, it only works with precomputed coefficient
 * => Tried this approach, very strict on coefficient input range.
 * => E.g a*b: a in [-R/2, R/2]. b in [-Q/2, Q/2] then c in [-2Q, 2Q]
 *
 *  - Barrett multiplication seem to work better with no restriction
 * => Proved to be good. E.g c=a*b, a in [-R, R], b in [-Q/2, Q/2] then c in [-3Q/2, 3Q/2]
 *
 * - Barrett reduction with c = a % Q. a in [-R, R] then c in [-Q/2, Q/2]
 *
 *
 * Doubling because
 * - Montgomery Doubling work with two unknown coefficient, no constaint at all
 * => c = a*b. a,b in [-R, R] c in [-Q, Q]
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

#define gsbf_mti(a, b, zl, zh, N, t, i) \
    t = vsubq_s16(a, b);                \
    a = vaddq_s16(a, b);                \
    b = vqrdmulhq_laneq_s16(t, zl, i);  \
    t = vmulq_laneq_s16(t, zh, i);      \
    b = vqrdmlahq_s16(b, t, N);

/*
 * GS Butterfly with Barrett *Rounding* reduction
 * Input: a in [-R, R], zl = w, zh = precomp_w, N, t
 * Output: c = a * b % Q. c in [-3Q/2, 3Q/2]
 */
#define gsbf_br(a, b, zl, zh, Q, t) \
    t = vsubq_s16(a, b);            \
    a = vaddq_s16(a, b);            \
    b = vmulq_s16(t, zl);           \
    t = vqrdmulhq_s16(t, zh);       \
    b = vmlsq_laneq_s16(b, t, Q, 0);

#define gsbf_bri(a, b, zl, zh, Q, t, i) \
    t = vsubq_s16(a, b);                \
    a = vaddq_s16(a, b);                \
    b = vmulq_laneq_s16(t, zl, i);      \
    t = vqrdmulhq_laneq_s16(t, zh, i);  \
    b = vmlsq_laneq_s16(b, t, Q, 0);

/*
 * Montgomery multiplication via *Rounding* use only for Inverse NTT
 * Input: a, b, bNinv, Q
 * Output: c = ab * R^-1
 */
#define montmul_invntt(a, zl, zh, Q, t, i) \
    a = vqrdmulhq_laneq_s16(a, zl, i);     \
    t = vmulq_laneq_s16(a, zh, i);         \
    a = vqrdmlahq_s16(a, t, Q);

/*
 * Barrett multiplication via *Rounding* use for Inverse NTT
 * Input: a, b, zl, zh, Q. a in [-R, R]
 * Output: c = a * b % Q. c in [-3Q/2, 3Q/2]
 */
#define barmul_invntt(a, zl, zh, Q, t, i) \
    t = vmulq_laneq_s16(a, zl, i);        \
    a = vqrdmulhq_laneq_s16(a, zh, i);    \
    a = vmlsq_laneq_s16(t, a, Q, 0);

#define barmul_invntt_x2(a, zl, zh, Q, t, i)              \
    t.val[0] = vmulq_laneq_s16(a.val[0], zl, i);          \
    t.val[1] = vmulq_laneq_s16(a.val[1], zl, i);          \
    a.val[0] = vqrdmulhq_laneq_s16(a.val[0], zh, i);      \
    a.val[1] = vqrdmulhq_laneq_s16(a.val[1], zh, i);      \
    a.val[0] = vmlsq_laneq_s16(t.val[0], a.val[0], Q, 0); \
    a.val[1] = vmlsq_laneq_s16(t.val[1], a.val[1], Q, 0);

#define barmul_invntt_x4(a, zl, zh, Q, t, i)              \
    t.val[0] = vmulq_laneq_s16(a.val[0], zl, i);          \
    t.val[1] = vmulq_laneq_s16(a.val[1], zl, i);          \
    t.val[2] = vmulq_laneq_s16(a.val[2], zl, i);          \
    t.val[3] = vmulq_laneq_s16(a.val[3], zl, i);          \
    a.val[0] = vqrdmulhq_laneq_s16(a.val[0], zh, i);      \
    a.val[1] = vqrdmulhq_laneq_s16(a.val[1], zh, i);      \
    a.val[2] = vqrdmulhq_laneq_s16(a.val[2], zh, i);      \
    a.val[3] = vqrdmulhq_laneq_s16(a.val[3], zh, i);      \
    a.val[0] = vmlsq_laneq_s16(t.val[0], a.val[0], Q, 0); \
    a.val[1] = vmlsq_laneq_s16(t.val[1], a.val[1], Q, 0); \
    a.val[2] = vmlsq_laneq_s16(t.val[2], a.val[2], Q, 0); \
    a.val[3] = vmlsq_laneq_s16(t.val[3], a.val[3], Q, 0);

#define barmuli_const(a, QMVM, t)        \
    t = vmulq_laneq_s16(a, QMVM, 2);     \
    a = vqrdmulhq_laneq_s16(a, QMVM, 6); \
    a = vmlsq_laneq_s16(t, a, QMVM, 0);

#define barmuli_const_x4(a, QMVM, t)                         \
    t.val[0] = vmulq_laneq_s16(a.val[0], QMVM, 2);           \
    t.val[1] = vmulq_laneq_s16(a.val[1], QMVM, 2);           \
    t.val[2] = vmulq_laneq_s16(a.val[2], QMVM, 2);           \
    t.val[3] = vmulq_laneq_s16(a.val[3], QMVM, 2);           \
    a.val[0] = vqrdmulhq_laneq_s16(a.val[0], QMVM, 6);       \
    a.val[1] = vqrdmulhq_laneq_s16(a.val[1], QMVM, 6);       \
    a.val[2] = vqrdmulhq_laneq_s16(a.val[2], QMVM, 6);       \
    a.val[3] = vqrdmulhq_laneq_s16(a.val[3], QMVM, 6);       \
    a.val[0] = vmlsq_laneq_s16(t.val[0], a.val[0], QMVM, 0); \
    a.val[1] = vmlsq_laneq_s16(t.val[1], a.val[1], QMVM, 0); \
    a.val[2] = vmlsq_laneq_s16(t.val[2], a.val[2], QMVM, 0); \
    a.val[3] = vmlsq_laneq_s16(t.val[3], a.val[3], QMVM, 0);

/*
 * CT Butterfly with Montgomery *Rounding* reduction
 * Input: a, b < R/2
 * Output: c in [-q, q], c = a * (2bR^-1)
 */
#define ctbf_mt(a, b, zl, zh, Q, t) \
    t = vqrdmulhq_s16(b, zl);       \
    b = vmulq_s16(b, zh);           \
    t = vqrdmlahq_s16(t, b, Q);     \
    b = vsubq_s16(a, t);            \
    a = vaddq_s16(a, t);

#define ctbf_mti(a, b, zl, zh, Q, t, i) \
    t = vqrdmulhq_laneq_s16(b, zl, i);  \
    b = vmulq_laneq_s16(b, zh, i);      \
    t = vqrdmlahq_s16(t, b, Q);         \
    b = vsubq_s16(a, t);                \
    a = vaddq_s16(a, t);

/*
 * CT Butterfly with Barrett *Rounding* reduction
 * Input: a in [-R, R], zl = w, zh = precomp_w, N, t
 * Output: c = a * b % Q. c in [-3Q/2, 3Q/2]
 */
#define ctbf_br(a, b, zl, zh, Q, t)  \
    t = vmulq_s16(b, zl);            \
    b = vqrdmulhq_s16(b, zh);        \
    t = vmlsq_laneq_s16(t, b, Q, 0); \
    b = vsubq_s16(a, t);             \
    a = vaddq_s16(a, t);

#define ctbf_bri(a, b, zl, zh, Q, t, i) \
    t = vmulq_laneq_s16(b, zl, i);      \
    b = vqrdmulhq_laneq_s16(b, zh, i);  \
    t = vmlsq_laneq_s16(t, b, Q, 0);    \
    b = vsubq_s16(a, t);                \
    a = vaddq_s16(a, t);

// ------------ Pointwise Multiplication ------------
/*
 * Montgomery multiplication via *Doubling*
 * Input: a, b, bNinv, Q
 * Output: c = ab * R^-1
 */
#define montmul(c, a, b, t, Q, Qinv) \
    c = vqdmulhq_s16(a, b);          \
    t = vmulq_s16(b, Qinv);          \
    t = vmulq_s16(a, t);             \
    t = vqdmulhq_s16(t, Q);          \
    c = vhsubq_u16(c, t);

// ------------ Barrett Reduction ------------
/*
 * Barrett reduction, return [-Q/2, Q/2]
 * `v` = 5461, `n` = 11
 */
#define barrett(a, t, QV)             \
    t = vqdmulhq_laneq_s16(a, QV, 4); \
    t = vrshrq_n_s16(t, 11);          \
    a = vmlsq_laneq_s16(a, t, QV, 0);

#define barrett_x4(a, t, QV)                               \
    t.val[0] = vqdmulhq_laneq_s16(a.val[0], QV, 4);        \
    t.val[1] = vqdmulhq_laneq_s16(a.val[1], QV, 4);        \
    t.val[2] = vqdmulhq_laneq_s16(a.val[2], QV, 4);        \
    t.val[3] = vqdmulhq_laneq_s16(a.val[3], QV, 4);        \
    t.val[0] = vrshrq_n_s16(t.val[0], 11);                 \
    t.val[1] = vrshrq_n_s16(t.val[1], 11);                 \
    t.val[2] = vrshrq_n_s16(t.val[2], 11);                 \
    t.val[3] = vrshrq_n_s16(t.val[3], 11);                 \
    a.val[0] = vmlsq_laneq_s16(a.val[0], t.val[0], QV, 0); \
    a.val[1] = vmlsq_laneq_s16(a.val[1], t.val[1], QV, 0); \
    a.val[2] = vmlsq_laneq_s16(a.val[2], t.val[2], QV, 0); \
    a.val[3] = vmlsq_laneq_s16(a.val[3], t.val[3], QV, 0);

#define barrett_x2(a, t, QV, i, j, m, n)                   \
    t.val[m] = vqdmulhq_laneq_s16(a.val[i], QV, 4);        \
    t.val[n] = vqdmulhq_laneq_s16(a.val[j], QV, 4);        \
    t.val[m] = vrshrq_n_s16(t.val[m], 11);                 \
    t.val[n] = vrshrq_n_s16(t.val[n], 11);                 \
    a.val[i] = vmlsq_laneq_s16(a.val[i], t.val[m], QV, 0); \
    a.val[j] = vmlsq_laneq_s16(a.val[j], t.val[n], QV, 0);

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

// ------------ Re-arrange vector ------------
#define arrange(v_out, v_in, i, j, m, n, a, b, c, d)                                      \
    v_out.val[a] = (int16x8_t)vtrn1q_s64((int64x2_t)v_in.val[i], (int64x2_t)v_in.val[j]); \
    v_out.val[b] = (int16x8_t)vtrn2q_s64((int64x2_t)v_in.val[i], (int64x2_t)v_in.val[j]); \
    v_out.val[c] = (int16x8_t)vtrn1q_s64((int64x2_t)v_in.val[m], (int64x2_t)v_in.val[n]); \
    v_out.val[d] = (int16x8_t)vtrn2q_s64((int64x2_t)v_in.val[m], (int64x2_t)v_in.val[n]);

#endif
