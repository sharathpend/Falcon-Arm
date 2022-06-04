/*
 * 64-bit Floating point NEON macro x1
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

#ifndef MACROF_H
#define MACROF_H

#include <arm_neon.h>
#include "config.h"

// c <= addr x1
#define vload(c, addr) c = vld1q_f64(addr);
// c <= addr interleave 2
#define vload2(c, addr) c = vld2q_f64(addr);
// c <= addr interleave 4
#define vload4(c, addr) c = vld4q_f64(addr);

#define vstore(addr, c) vst1q_f64(addr, c);
// addr <= c
#define vstore2(addr, c) vst2q_f64(addr, c);
// addr <= c
#define vstore4(addr, c) vst4q_f64(addr, c);

// c <= addr x2
#define vloadx2(c, addr) c = vld1q_f64_x2(addr);
// c <= addr x3
#define vloadx3(c, addr) c = vld1q_f64_x3(addr);

// addr <= c
#define vstorex2(addr, c) vst1q_f64_x2(addr, c);

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

// c = a * b[i]
#define vfmul_lane(c, a, b, i) c = vmulq_laneq_f64(a, b, i);

// c = 1/a
#define vfinv(c, a) c = vdivq_f64(vdupq_n_f64(1.0), a);

// c = -a
#define vfneg(c, a) c = vnegq_f64(a);

#define transpose(a, b, t, ia, ib, it)            \
    t.val[it] = a.val[ia];                        \
    a.val[ia] = vzip1q_f64(a.val[ia], b.val[ib]); \
    b.val[ib] = vzip2q_f64(t.val[it], b.val[ib]);

/*
 * c = a + jb
 * c[0] = a[0] - b[1]
 * c[1] = a[1] + b[0]
 */
#define vfcaddj(c, a, b) c = vcaddq_rot90_f64(a, b);

/*
 * c = a - jb
 * c[0] = a[0] + b[1]
 * c[1] = a[1] - b[0]
 */
#define vfcsubj(c, a, b) c = vcaddq_rot270_f64(a, b);

// c[0] = c[0] + b[0]*a[0], c[1] = c[1] + b[1]*a[0]
#define vfcmla(c, a, b) c = vcmlaq_f64(c, a, b);

// c[0] = c[0] - b[1]*a[1], c[1] = c[1] + b[0]*a[1]
#define vfcmla_90(c, a, b) c = vcmlaq_rot90_f64(c, a, b);

// c[0] = c[0] - b[0]*a[0], c[1] = c[1] - b[1]*a[0]
#define vfcmla_180(c, a, b) c = vcmlaq_rot180_f64(c, a, b);

// c[0] = c[0] + b[1]*a[1], c[1] = c[1] - b[0]*a[1]
#define vfcmla_270(c, a, b) c = vcmlaq_rot270_f64(c, a, b);

/*
 * Complex MUL: c = a*b
 * c[0] = a[0]*b[0] - a[1]*b[1]
 * c[1] = a[0]*b[1] + a[1]*b[0]
 */
// TODO: rename this macro
#define FPC_MUL(c, a, b)          \
    c = vmulq_laneq_f64(b, a, 0); \
    c = vcmlaq_rot90_f64(c, a, b);

#define FPC_CMUL(c, a, b)         \
    c = vmulq_laneq_f64(b, a, 0); \
    c = vcmlaq_rot90_f64(c, a, b);

/*
 * Complex MUL: c = a * conjugate(b) = a * (b[0], -b[1])
 * c[0] =   b[0]*a[0] + b[1]*a[1]
 * c[1] = + b[0]*a[1] - b[1]*a[0]
 */
#define FPC_MULCONJ(c, a, b)      \
    c = vmulq_laneq_f64(a, b, 0); \
    c = vcmlaq_rot270_f64(c, b, a);

#define FPC_ADDJ(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vsubq_f64(a_re, b_im);                    \
    d_im = vaddq_f64(a_im, b_re);

#define FPC_SUBJ(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vaddq_f64(a_re, b_im);                    \
    d_im = vsubq_f64(a_im, b_re);

#define FPC_SUB(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vsubq_f64(a_re, b_re);                   \
    d_im = vsubq_f64(a_im, b_im);

#define FPC_SUBx2(d_re, d_im, a_re, a_im, b_re, b_im, d0, t0) \
    d_re.val[d0] = vsubq_f64(a_re.val[t0], b_re.val[t0]);     \
    d_im.val[d0] = vsubq_f64(a_im.val[t0], b_im.val[t0]);

#define FPC_ADD(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vaddq_f64(a_re, b_re);                   \
    d_im = vaddq_f64(a_im, b_im);

#define FPC_ADDx2(d_re, d_im, a_re, a_im, b_re, b_im, d0, t0) \
    d_re.val[d0] = vaddq_f64(a_re.val[t0], b_re.val[t0]);     \
    d_im.val[d0] = vaddq_f64(a_im.val[t0], b_im.val[t0]);

#define FPC_SUBx4(d_re, d_im, a_re, a_im, b_re, b_im)  \
    d_re.val[0] = vsubq_f64(a_re.val[0], b_re.val[0]); \
    d_im.val[0] = vsubq_f64(a_im.val[0], b_im.val[0]); \
    d_re.val[1] = vsubq_f64(a_re.val[1], b_re.val[1]); \
    d_im.val[1] = vsubq_f64(a_im.val[1], b_im.val[1]); \
    d_re.val[2] = vsubq_f64(a_re.val[2], b_re.val[2]); \
    d_im.val[2] = vsubq_f64(a_im.val[2], b_im.val[2]); \
    d_re.val[3] = vsubq_f64(a_re.val[3], b_re.val[3]); \
    d_im.val[3] = vsubq_f64(a_im.val[3], b_im.val[3]);

#define FPC_ADDx4(d_re, d_im, a_re, a_im, b_re, b_im)  \
    d_re.val[0] = vaddq_f64(a_re.val[0], b_re.val[0]); \
    d_im.val[0] = vaddq_f64(a_im.val[0], b_im.val[0]); \
    d_re.val[1] = vaddq_f64(a_re.val[1], b_re.val[1]); \
    d_im.val[1] = vaddq_f64(a_im.val[1], b_im.val[1]); \
    d_re.val[2] = vaddq_f64(a_re.val[2], b_re.val[2]); \
    d_im.val[2] = vaddq_f64(a_im.val[2], b_im.val[2]); \
    d_re.val[3] = vaddq_f64(a_re.val[3], b_re.val[3]); \
    d_im.val[3] = vaddq_f64(a_im.val[3], b_im.val[3]);

#define FPC_MUL_LANE(d_re, d_im, a_re, a_im, b_re_im) \
    d_re = vmulq_laneq_f64(a_re, b_re_im, 0);         \
    d_re = vfmsq_laneq_f64(d_re, a_im, b_re_im, 1);   \
    d_im = vmulq_laneq_f64(a_re, b_re_im, 1);         \
    d_im = vfmaq_laneq_f64(d_im, a_im, b_re_im, 0);

#define FPC_MUL_LANEx4(d_re, d_im, a_re, a_im, b_re_im)                  \
    d_re.val[0] = vmulq_laneq_f64(a_re.val[0], b_re_im, 0);              \
    d_re.val[0] = vfmsq_laneq_f64(d_re.val[0], a_im.val[0], b_re_im, 1); \
    d_re.val[1] = vmulq_laneq_f64(a_re.val[1], b_re_im, 0);              \
    d_re.val[1] = vfmsq_laneq_f64(d_re.val[1], a_im.val[1], b_re_im, 1); \
    d_re.val[2] = vmulq_laneq_f64(a_re.val[2], b_re_im, 0);              \
    d_re.val[2] = vfmsq_laneq_f64(d_re.val[2], a_im.val[2], b_re_im, 1); \
    d_re.val[3] = vmulq_laneq_f64(a_re.val[3], b_re_im, 0);              \
    d_re.val[3] = vfmsq_laneq_f64(d_re.val[3], a_im.val[3], b_re_im, 1); \
    d_im.val[0] = vmulq_laneq_f64(a_re.val[0], b_re_im, 1);              \
    d_im.val[0] = vfmaq_laneq_f64(d_im.val[0], a_im.val[0], b_re_im, 0); \
    d_im.val[1] = vmulq_laneq_f64(a_re.val[1], b_re_im, 1);              \
    d_im.val[1] = vfmaq_laneq_f64(d_im.val[1], a_im.val[1], b_re_im, 0); \
    d_im.val[2] = vmulq_laneq_f64(a_re.val[2], b_re_im, 1);              \
    d_im.val[2] = vfmaq_laneq_f64(d_im.val[2], a_im.val[2], b_re_im, 0); \
    d_im.val[3] = vmulq_laneq_f64(a_re.val[3], b_re_im, 1);              \
    d_im.val[3] = vfmaq_laneq_f64(d_im.val[3], a_im.val[3], b_re_im, 0);

#define FPC_MUL_LANEx2(d_re, d_im, a_re, a_im, b_re_im, d0, d1, i0, i1)     \
    d_re.val[d0] = vmulq_laneq_f64(a_re.val[i0], b_re_im, 0);               \
    d_re.val[d0] = vfmsq_laneq_f64(d_re.val[i0], a_im.val[i0], b_re_im, 1); \
    d_re.val[d1] = vmulq_laneq_f64(a_re.val[i1], b_re_im, 0);               \
    d_re.val[d1] = vfmsq_laneq_f64(d_re.val[i1], a_im.val[i1], b_re_im, 1); \
    d_im.val[d0] = vmulq_laneq_f64(a_re.val[i0], b_re_im, 1);               \
    d_im.val[d0] = vfmaq_laneq_f64(d_im.val[i0], a_im.val[i0], b_re_im, 0); \
    d_im.val[d1] = vmulq_laneq_f64(a_re.val[i1], b_re_im, 1);               \
    d_im.val[d1] = vfmaq_laneq_f64(d_im.val[i1], a_im.val[i1], b_re_im, 0);

#define FPC_MUL_1(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vmulq_f64(a_re, b_re);                     \
    d_re = vfmsq_f64(d_re, a_im, b_im);               \
    d_im = vmulq_f64(a_re, b_im);                     \
    d_im = vfmaq_f64(d_im, a_im, b_re);

#define FWD_BUF(a_re, a_im, b_re, b_im, zeta, t_re, t_im) \
    FPC_MUL_LANE(t_re, t_im, b_re, b_im, zeta);           \
    FPC_SUB(b_re, b_im, a_re, a_im, t_re, t_im);          \
    FPC_ADD(a_re, a_im, a_re, a_im, t_re, t_im);

#define FWD_TOP_LANE(t_re, t_im, b_re, b_im, zeta) \
    FPC_MUL_LANE(t_re, t_im, b_re, b_im, zeta);

#define FWD_TOP_LANEx4(t_re, t_im, b_re, b_im, zeta) \
    FPC_MUL_LANEx4(t_re, t_im, b_re, b_im, zeta);

#define FWD_TOP_LANEx2(t_re, t_im, a_re, a_im, b_re_im, d0, d1, i0, i1) \
    FPC_MUL_LANEx2(t_re, t_im, a_re, a_im, b_re_im, d0, d1, i0, i1);

#define FWD_TOP(t_re, t_im, b_re, b_im, zeta_re, zeta_im) \
    FPC_MUL_1(t_re, t_im, b_re, b_im, zeta_re, zeta_im);

#define FWD_TOPx4(t_re, t_im, b_re, b_im, zeta_re, zeta_im) \
    FPC_MUL_4(t_re, t_im, b_re, b_im, zeta_re, zeta_im);

#define FWD_BOT(a_re, a_im, b_re, b_im, t_re, t_im) \
    FPC_SUB(b_re, b_im, a_re, a_im, t_re, t_im);    \
    FPC_ADD(a_re, a_im, a_re, a_im, t_re, t_im);

#define FWD_BOTx4(a_re, a_im, b_re, b_im, t_re, t_im) \
    FPC_SUBx4(b_re, b_im, a_re, a_im, t_re, t_im);    \
    FPC_ADDx4(a_re, a_im, a_re, a_im, t_re, t_im);

#define FWD_BOTJ(a_re, a_im, b_re, b_im, t_re, t_im) \
    FPC_SUBJ(b_re, b_im, a_re, a_im, t_re, t_im);    \
    FPC_ADDJ(a_re, a_im, a_re, a_im, t_re, t_im);

//============== Inverse FFT
/*
 * a * conj(b)
 * Original (without swap):
 * d_re = b_im * a_im + a_re * b_re;
 * d_im = b_re * a_im - a_re * b_im;
 */
#define FPC_MUL_LANE_1(d_re, d_im, a_re, a_im, b_re_im) \
    d_re = vmulq_laneq_f64(a_re, b_re_im, 0);           \
    d_re = vfmaq_laneq_f64(d_re, a_im, b_re_im, 1);     \
    d_im = vmulq_laneq_f64(a_im, b_re_im, 0);           \
    d_im = vfmsq_laneq_f64(d_im, a_re, b_re_im, 1);

#define FPC_MUL_LANE_1x4(d_re, d_im, a_re, a_im, b_re_im)                \
    d_re.val[0] = vmulq_laneq_f64(a_re.val[0], b_re_im, 0);              \
    d_re.val[0] = vfmaq_laneq_f64(d_re.val[0], a_im.val[0], b_re_im, 1); \
    d_im.val[0] = vmulq_laneq_f64(a_im.val[0], b_re_im, 0);              \
    d_im.val[0] = vfmsq_laneq_f64(d_im.val[0], a_re.val[0], b_re_im, 1); \
    d_re.val[1] = vmulq_laneq_f64(a_re.val[1], b_re_im, 0);              \
    d_re.val[1] = vfmaq_laneq_f64(d_re.val[1], a_im.val[1], b_re_im, 1); \
    d_im.val[1] = vmulq_laneq_f64(a_im.val[1], b_re_im, 0);              \
    d_im.val[1] = vfmsq_laneq_f64(d_im.val[1], a_re.val[1], b_re_im, 1); \
    d_re.val[2] = vmulq_laneq_f64(a_re.val[2], b_re_im, 0);              \
    d_re.val[2] = vfmaq_laneq_f64(d_re.val[2], a_im.val[2], b_re_im, 1); \
    d_im.val[2] = vmulq_laneq_f64(a_im.val[2], b_re_im, 0);              \
    d_im.val[2] = vfmsq_laneq_f64(d_im.val[2], a_re.val[2], b_re_im, 1); \
    d_re.val[3] = vmulq_laneq_f64(a_re.val[3], b_re_im, 0);              \
    d_re.val[3] = vfmaq_laneq_f64(d_re.val[3], a_im.val[3], b_re_im, 1); \
    d_im.val[3] = vmulq_laneq_f64(a_im.val[3], b_re_im, 0);              \
    d_im.val[3] = vfmsq_laneq_f64(d_im.val[3], a_re.val[3], b_re_im, 1);

#define FPC_MUL_2(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vmulq_f64(b_im, a_im);                     \
    d_re = vfmaq_f64(d_re, a_re, b_re);               \
    d_im = vmulq_f64(b_re, a_im);                     \
    d_im = vfmsq_f64(d_im, a_re, b_im);

/*
 * a * -conj(b)
 * d_re = a_re * b_im - a_im * b_re;
 * d_im = a_im * b_im + a_re * b_re;
 */
#define FPC_MUL_LANE_2(d_re, d_im, a_re, a_im, b_re_im) \
    d_re = vmulq_laneq_f64(a_re, b_re_im, 1);           \
    d_re = vfmsq_laneq_f64(d_re, a_im, b_re_im, 0);     \
    d_im = vmulq_laneq_f64(a_re, b_re_im, 0);           \
    d_im = vfmaq_laneq_f64(d_im, a_im, b_re_im, 1);

#define FPC_MUL_LANE_2x4(d_re, d_im, a_re, a_im, b_re_im)                \
    d_re.val[0] = vmulq_laneq_f64(a_re.val[0], b_re_im, 1);              \
    d_re.val[0] = vfmsq_laneq_f64(d_re.val[0], a_im.val[0], b_re_im, 0); \
    d_im.val[0] = vmulq_laneq_f64(a_re.val[0], b_re_im, 0);              \
    d_im.val[0] = vfmaq_laneq_f64(d_im.val[0], a_im.val[0], b_re_im, 1); \
    d_re.val[1] = vmulq_laneq_f64(a_re.val[1], b_re_im, 1);              \
    d_re.val[1] = vfmsq_laneq_f64(d_re.val[1], a_im.val[1], b_re_im, 0); \
    d_im.val[1] = vmulq_laneq_f64(a_re.val[1], b_re_im, 0);              \
    d_im.val[1] = vfmaq_laneq_f64(d_im.val[1], a_im.val[1], b_re_im, 1); \
    d_re.val[2] = vmulq_laneq_f64(a_re.val[2], b_re_im, 1);              \
    d_re.val[2] = vfmsq_laneq_f64(d_re.val[2], a_im.val[2], b_re_im, 0); \
    d_im.val[2] = vmulq_laneq_f64(a_re.val[2], b_re_im, 0);              \
    d_im.val[2] = vfmaq_laneq_f64(d_im.val[2], a_im.val[2], b_re_im, 1); \
    d_re.val[3] = vmulq_laneq_f64(a_re.val[3], b_re_im, 1);              \
    d_re.val[3] = vfmsq_laneq_f64(d_re.val[3], a_im.val[3], b_re_im, 0); \
    d_im.val[3] = vmulq_laneq_f64(a_re.val[3], b_re_im, 0);              \
    d_im.val[3] = vfmaq_laneq_f64(d_im.val[3], a_im.val[3], b_re_im, 1);

#define FPC_MUL_3(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vmulq_f64(a_re, b_im);                     \
    d_re = vfmsq_f64(d_re, a_im, b_re);               \
    d_im = vmulq_f64(a_im, b_im);                     \
    d_im = vfmaq_f64(d_im, a_re, b_re);

#define INV_TOPJ(t_re, t_im, a_re, a_im, b_re, b_im) \
    FPC_SUB(t_re, t_im, a_re, a_im, b_re, b_im);     \
    FPC_ADD(a_re, a_im, a_re, a_im, b_re, b_im);

#define INV_TOPJx4(t_re, t_im, a_re, a_im, b_re, b_im) \
    FPC_SUBx4(t_re, t_im, a_re, a_im, b_re, b_im);     \
    FPC_ADDx4(a_re, a_im, a_re, a_im, b_re, b_im);

#define INV_BOTJ_LANE(b_re, b_im, t_re, t_im, zeta) \
    FPC_MUL_LANE_1(b_re, b_im, t_re, t_im, zeta);

#define INV_BOTJ_LANEx4(b_re, b_im, t_re, t_im, zeta) \
    FPC_MUL_LANE_1x4(b_re, b_im, t_re, t_im, zeta);

#define INV_BOTJ(b_re, b_im, t_re, t_im, zeta_re, zeta_im) \
    FPC_MUL_2(b_re, b_im, t_re, t_im, zeta_re, zeta_im);

#define INV_TOPJm(t_re, t_im, a_re, a_im, b_re, b_im) \
    FPC_SUB(t_re, t_im, b_re, b_im, a_re, a_im);      \
    FPC_ADD(a_re, a_im, a_re, a_im, b_re, b_im);

#define INV_TOPJmx4(t_re, t_im, a_re, a_im, b_re, b_im) \
    FPC_SUBx4(t_re, t_im, b_re, b_im, a_re, a_im);      \
    FPC_ADDx4(a_re, a_im, a_re, a_im, b_re, b_im);

#define INV_BOTJm_LANE(b_re, b_im, t_re, t_im, zeta) \
    FPC_MUL_LANE_2(b_re, b_im, t_re, t_im, zeta);

#define INV_BOTJm_LANEx4(b_re, b_im, t_re, t_im, zeta) \
    FPC_MUL_LANE_2x4(b_re, b_im, t_re, t_im, zeta);

#define INV_BOTJm(b_re, b_im, t_re, t_im, zeta_re, zeta_im) \
    FPC_MUL_3(b_re, b_im, t_re, t_im, zeta_re, zeta_im);

#if FMA == 1
// d = c + a *b
#define vfma(d, c, a, b) d = vfmaq_f64(c, a, b); // 7 cycles
// d = c - a * b
#define vfms(d, c, a, b) d = vfmsq_f64(c, a, b);
// d = c + a * b[i]
#define vfma_lane(d, c, a, b, i) d = vfmaq_laneq_f64(c, a, b, i);
// d = c - a * b[i]
#define vfms_lane(d, c, a, b, i) d = vfmsq_laneq_f64(c, a, b, i);

#else
// d = c + a *b
#define vfma(d, c, a, b) d = vaddq_f64(c, vmulq_f64(a, b)); // 8 cycles
// d = c - a * b
#define vfms(d, c, a, b) d = vsubq_f64(c, vmulq_f64(a, b));
// d = c + a * b[i]
#define vfma_lane(d, c, a, b, i) d = vaddq_f64(c, vmulq_laneq_f64(a, b, i));
// d = c - a * b[i]
#define vfms_lane(d, c, a, b, i) d = vsubq_f64(c, vmulq_laneq_f64(a, b, i));
#endif

#endif

// I add this
#define FPC_MULx2(d_re, d_im, a_re, a_im, b_re, b_im)               \
    d_re.val[0] = vmulq_f64(a_re.val[0], b_re.val[0]);              \
    d_re.val[0] = vfmsq_f64(d_re.val[0], a_im.val[0], b_im.val[0]); \
    d_re.val[1] = vmulq_f64(a_re.val[1], b_re.val[1]);              \
    d_re.val[1] = vfmsq_f64(d_re.val[1], a_im.val[1], b_im.val[1]); \
    d_im.val[0] = vmulq_f64(a_re.val[0], b_im.val[0]);              \
    d_im.val[0] = vfmaq_f64(d_im.val[0], a_im.val[0], b_re.val[0]); \
    d_im.val[1] = vmulq_f64(a_re.val[1], b_im.val[1]);              \
    d_im.val[1] = vfmaq_f64(d_im.val[1], a_im.val[1], b_re.val[1]);

#define FPC_MULx4(d_re, d_im, a_re, a_im, b_re, b_im)               \
    d_re.val[0] = vmulq_f64(a_re.val[0], b_re.val[0]);              \
    d_re.val[0] = vfmsq_f64(d_re.val[0], a_im.val[0], b_im.val[0]); \
    d_re.val[1] = vmulq_f64(a_re.val[1], b_re.val[1]);              \
    d_re.val[1] = vfmsq_f64(d_re.val[1], a_im.val[1], b_im.val[1]); \
    d_re.val[2] = vmulq_f64(a_re.val[2], b_re.val[2]);              \
    d_re.val[2] = vfmsq_f64(d_re.val[2], a_im.val[2], b_im.val[2]); \
    d_re.val[3] = vmulq_f64(a_re.val[3], b_re.val[3]);              \
    d_re.val[3] = vfmsq_f64(d_re.val[3], a_im.val[3], b_im.val[3]); \
    d_im.val[0] = vmulq_f64(a_re.val[0], b_im.val[0]);              \
    d_im.val[0] = vfmaq_f64(d_im.val[0], a_im.val[0], b_re.val[0]); \
    d_im.val[1] = vmulq_f64(a_re.val[1], b_im.val[1]);              \
    d_im.val[1] = vfmaq_f64(d_im.val[1], a_im.val[1], b_re.val[1]); \
    d_im.val[2] = vmulq_f64(a_re.val[2], b_im.val[2]);              \
    d_im.val[2] = vfmaq_f64(d_im.val[2], a_im.val[2], b_re.val[2]); \
    d_im.val[3] = vmulq_f64(a_re.val[3], b_im.val[3]);              \
    d_im.val[3] = vfmaq_f64(d_im.val[3], a_im.val[3], b_re.val[3]);

#define FPC_MLA(d_re, d_im, a_re, a_im, b_re, b_im) \
    d_re = vfmaq_f64(d_re, a_re, b_re);             \
    d_re = vfmsq_f64(d_re, a_im, b_im);             \
    d_im = vfmaq_f64(d_im, a_re, b_im);             \
    d_im = vfmaq_f64(d_im, a_im, b_re);

#define FPC_MLAx2(d_re, d_im, a_re, a_im, b_re, b_im)               \
    d_re.val[0] = vfmaq_f64(d_re.val[0], a_re.val[0], b_re.val[0]); \
    d_re.val[0] = vfmsq_f64(d_re.val[0], a_im.val[0], b_im.val[0]); \
    d_re.val[1] = vfmaq_f64(d_re.val[1], a_re.val[1], b_re.val[1]); \
    d_re.val[1] = vfmsq_f64(d_re.val[1], a_im.val[1], b_im.val[1]); \
    d_im.val[0] = vfmaq_f64(d_im.val[0], a_re.val[0], b_im.val[0]); \
    d_im.val[0] = vfmaq_f64(d_im.val[0], a_im.val[0], b_re.val[0]); \
    d_im.val[1] = vfmaq_f64(d_im.val[1], a_re.val[1], b_im.val[1]); \
    d_im.val[1] = vfmaq_f64(d_im.val[1], a_im.val[1], b_re.val[1]);

#define FPC_MLAx4(d_re, d_im, a_re, a_im, b_re, b_im)               \
    d_re.val[0] = vfmaq_f64(d_re.val[0], a_re.val[0], b_re.val[0]); \
    d_re.val[0] = vfmsq_f64(d_re.val[0], a_im.val[0], b_im.val[0]); \
    d_re.val[1] = vfmaq_f64(d_re.val[1], a_re.val[1], b_re.val[1]); \
    d_re.val[1] = vfmsq_f64(d_re.val[1], a_im.val[1], b_im.val[1]); \
    d_re.val[2] = vfmaq_f64(d_re.val[2], a_re.val[2], b_re.val[2]); \
    d_re.val[2] = vfmsq_f64(d_re.val[2], a_im.val[2], b_im.val[2]); \
    d_re.val[3] = vfmaq_f64(d_re.val[3], a_re.val[3], b_re.val[3]); \
    d_re.val[3] = vfmsq_f64(d_re.val[3], a_im.val[3], b_im.val[3]); \
    d_im.val[0] = vfmaq_f64(d_im.val[0], a_re.val[0], b_im.val[0]); \
    d_im.val[0] = vfmaq_f64(d_im.val[0], a_im.val[0], b_re.val[0]); \
    d_im.val[1] = vfmaq_f64(d_im.val[1], a_re.val[1], b_im.val[1]); \
    d_im.val[1] = vfmaq_f64(d_im.val[1], a_im.val[1], b_re.val[1]); \
    d_im.val[2] = vfmaq_f64(d_im.val[2], a_re.val[2], b_im.val[2]); \
    d_im.val[2] = vfmaq_f64(d_im.val[2], a_im.val[2], b_re.val[2]); \
    d_im.val[3] = vfmaq_f64(d_im.val[3], a_re.val[3], b_im.val[3]); \
    d_im.val[3] = vfmaq_f64(d_im.val[3], a_im.val[3], b_re.val[3]);

#define FPC_MUL_CONJx4(d_re, d_im, a_re, a_im, b_re, b_im)          \
    d_re.val[0] = vmulq_f64(b_im.val[0], a_im.val[0]);              \
    d_re.val[0] = vfmaq_f64(d_re.val[0], a_re.val[0], b_re.val[0]); \
    d_re.val[1] = vmulq_f64(b_im.val[1], a_im.val[1]);              \
    d_re.val[1] = vfmaq_f64(d_re.val[1], a_re.val[1], b_re.val[1]); \
    d_re.val[2] = vmulq_f64(b_im.val[2], a_im.val[2]);              \
    d_re.val[2] = vfmaq_f64(d_re.val[2], a_re.val[2], b_re.val[2]); \
    d_re.val[3] = vmulq_f64(b_im.val[3], a_im.val[3]);              \
    d_re.val[3] = vfmaq_f64(d_re.val[3], a_re.val[3], b_re.val[3]); \
    d_im.val[0] = vmulq_f64(b_re.val[0], a_im.val[0]);              \
    d_im.val[0] = vfmsq_f64(d_im.val[0], a_re.val[0], b_im.val[0]); \
    d_im.val[1] = vmulq_f64(b_re.val[1], a_im.val[1]);              \
    d_im.val[1] = vfmsq_f64(d_im.val[1], a_re.val[1], b_im.val[1]); \
    d_im.val[2] = vmulq_f64(b_re.val[2], a_im.val[2]);              \
    d_im.val[2] = vfmsq_f64(d_im.val[2], a_re.val[2], b_im.val[2]); \
    d_im.val[3] = vmulq_f64(b_re.val[3], a_im.val[3]);              \
    d_im.val[3] = vfmsq_f64(d_im.val[3], a_re.val[3], b_im.val[3]);

#define FPC_MLA_CONJx4(d_re, d_im, a_re, a_im, b_re, b_im)          \
    d_re.val[0] = vfmaq_f64(d_re.val[0], b_im.val[0], a_im.val[0]); \
    d_re.val[0] = vfmaq_f64(d_re.val[0], a_re.val[0], b_re.val[0]); \
    d_re.val[1] = vfmaq_f64(d_re.val[1], b_im.val[1], a_im.val[1]); \
    d_re.val[1] = vfmaq_f64(d_re.val[1], a_re.val[1], b_re.val[1]); \
    d_re.val[2] = vfmaq_f64(d_re.val[2], b_im.val[2], a_im.val[2]); \
    d_re.val[2] = vfmaq_f64(d_re.val[2], a_re.val[2], b_re.val[2]); \
    d_re.val[3] = vfmaq_f64(d_re.val[3], b_im.val[3], a_im.val[3]); \
    d_re.val[3] = vfmaq_f64(d_re.val[3], a_re.val[3], b_re.val[3]); \
    d_im.val[0] = vfmaq_f64(d_im.val[0], b_re.val[0], a_im.val[0]); \
    d_im.val[0] = vfmsq_f64(d_im.val[0], a_re.val[0], b_im.val[0]); \
    d_im.val[1] = vfmaq_f64(d_im.val[1], b_re.val[1], a_im.val[1]); \
    d_im.val[1] = vfmsq_f64(d_im.val[1], a_re.val[1], b_im.val[1]); \
    d_im.val[2] = vfmaq_f64(d_im.val[2], b_re.val[2], a_im.val[2]); \
    d_im.val[2] = vfmsq_f64(d_im.val[2], a_re.val[2], b_im.val[2]); \
    d_im.val[3] = vfmaq_f64(d_im.val[3], b_re.val[3], a_im.val[3]); \
    d_im.val[3] = vfmsq_f64(d_im.val[3], a_re.val[3], b_im.val[3]);
