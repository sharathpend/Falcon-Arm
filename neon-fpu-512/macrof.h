#ifndef MACROF_H
#define MACROF_H

#include <arm_neon.h>

// c <= addr x1
#define vload(c, addr) c = vld1q_f64(addr);
// c <= addr interleave 2
#define vload2(c, addr) c = vld2q_f64(addr);
// c <= addr interleave 4
#define vload4(c, addr) c = vld4q_f64(addr);

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

#endif
