/*
 * 64-bit Floating point NEON macro x4
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

#ifndef MACROFX4_H
#define MACROFX4_H

#include <arm_neon.h>
#include "config.h"

#define vloadx4(c, addr) c = vld1q_f64_x4(addr);

#define vstorex4(addr, c) vst1q_f64_x4(addr, c);

#define vfdupx4(c, constant)   \
    c.val[0] = vdupq_n_f64(0); \
    c.val[1] = vdupq_n_f64(0); \
    c.val[2] = vdupq_n_f64(0); \
    c.val[3] = vdupq_n_f64(0);

#define vfnegx4(c, a)               \
    c.val[0] = vnegq_f64(a.val[0]); \
    c.val[1] = vnegq_f64(a.val[1]); \
    c.val[2] = vnegq_f64(a.val[2]); \
    c.val[3] = vnegq_f64(a.val[3]);

#define vfmulnx4(c, a, n)                \
    c.val[0] = vmulq_n_f64(a.val[0], n); \
    c.val[1] = vmulq_n_f64(a.val[1], n); \
    c.val[2] = vmulq_n_f64(a.val[2], n); \
    c.val[3] = vmulq_n_f64(a.val[3], n);

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

#define vfmulx4(c, a, b)                      \
    c.val[0] = vmulq_f64(a.val[0], b.val[0]); \
    c.val[1] = vmulq_f64(a.val[1], b.val[1]); \
    c.val[2] = vmulq_f64(a.val[2], b.val[2]); \
    c.val[3] = vmulq_f64(a.val[3], b.val[3]);

#define vfmulx4_lane(c, a, b, i)                \
    c.val[0] = vmulq_laneq_f64(a.val[0], b, i); \
    c.val[1] = vmulq_laneq_f64(a.val[1], b, i); \
    c.val[2] = vmulq_laneq_f64(a.val[2], b, i); \
    c.val[3] = vmulq_laneq_f64(a.val[3], b, i);

#define vfdivx4(c, a, b)                      \
    c.val[0] = vdivq_f64(a.val[0], b.val[0]); \
    c.val[1] = vdivq_f64(a.val[0], b.val[1]); \
    c.val[2] = vdivq_f64(a.val[0], b.val[2]); \
    c.val[3] = vdivq_f64(a.val[0], b.val[3]);

#define vfinvx4(c, a)                                 \
    c.val[0] = vdivq_f64(vdupq_n_f64(1.0), a.val[0]); \
    c.val[1] = vdivq_f64(vdupq_n_f64(1.0), a.val[1]); \
    c.val[2] = vdivq_f64(vdupq_n_f64(1.0), a.val[2]); \
    c.val[3] = vdivq_f64(vdupq_n_f64(1.0), a.val[3]);

#define vfcvtx4(c, a)                   \
    c.val[0] = vcvtq_f64_s64(a.val[0]); \
    c.val[1] = vcvtq_f64_s64(a.val[1]); \
    c.val[2] = vcvtq_f64_s64(a.val[2]); \
    c.val[3] = vcvtq_f64_s64(a.val[3]);

#if FMA == 1
#define vfmax4(d, c, a, b)                              \
    d.val[0] = vfmaq_f64(c.val[0], a.val[0], b.val[0]); \
    d.val[1] = vfmaq_f64(c.val[1], a.val[1], b.val[1]); \
    d.val[2] = vfmaq_f64(c.val[2], a.val[2], b.val[2]); \
    d.val[3] = vfmaq_f64(c.val[3], a.val[3], b.val[3]);
#define vfmsx4(d, c, a, b)                              \
    d.val[0] = vfmsq_f64(c.val[0], a.val[0], b.val[0]); \
    d.val[1] = vfmsq_f64(c.val[1], a.val[1], b.val[1]); \
    d.val[2] = vfmsq_f64(c.val[2], a.val[2], b.val[2]); \
    d.val[3] = vfmsq_f64(c.val[3], a.val[3], b.val[3]);
#else
#define vfmax4(d, c, a, b)                                         \
    d.val[0] = vaddq_f64(c.val[0], vmulq_f64(a.val[0], b.val[0])); \
    d.val[1] = vaddq_f64(c.val[1], vmulq_f64(a.val[1], b.val[1])); \
    d.val[2] = vaddq_f64(c.val[2], vmulq_f64(a.val[2], b.val[2])); \
    d.val[3] = vaddq_f64(c.val[3], vmulq_f64(a.val[3], b.val[3]));

#define vfmsx4(d, c, a, b)                                         \
    d.val[0] = vsubq_f64(c.val[0], vmulq_f64(a.val[0], b.val[0])); \
    d.val[1] = vsubq_f64(c.val[1], vmulq_f64(a.val[1], b.val[1])); \
    d.val[2] = vsubq_f64(c.val[2], vmulq_f64(a.val[2], b.val[2])); \
    d.val[3] = vsubq_f64(c.val[3], vmulq_f64(a.val[3], b.val[3]));
#endif

#endif
