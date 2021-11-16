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
    a.val[ia] = vzip1q_f64(t.val[it], b.val[ib]); \
    b.val[ib] = vzip2q_f64(t.val[it], b.val[ib]);


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
