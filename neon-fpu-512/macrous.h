#ifndef MACROUS_H
#define MACROUS_H

#include <arm_neon.h>

#define vmull_lo(c, a, b) c = vmull_s16(vget_low_s16(a), vget_low_s16(b));

#define vmull_hi(c, a, b) c = vmull_high_s16(a, b);

#define vmulla_lo(d, c, a, b) d = vmlal_u16(c, (uint16x4_t)vget_low_s16(a), (uint16x4_t)vget_low_s16(b));

#define vmulla_hi(d, c, a, b) d = vmlal_high_u16(c, (uint16x8_t)a, (uint16x8_t)b);

#define vadd(c, a, b) c = vaddq_u32(a, b);

#define vaddv(c, a) c = vaddvq_u32(a);

#define vor(c, a, b) c = vorrq_u32(a, b);

#define vdup32x4(c, constant)         \
    c.val[0] = vdupq_n_u32(constant); \
    c.val[1] = vdupq_n_u32(constant); \
    c.val[2] = vdupq_n_u32(constant); \
    c.val[3] = vdupq_n_u32(constant);

#endif
