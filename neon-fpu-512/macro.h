#ifndef MACRO_H
#define MACRO_H

#include <arm_neon.h>

#define vmull_lo(c, a, b) c = vmull_s16(vget_low_s16(a), vget_low_s16(b));

#define vmull_hi(c, a, b) c = vmull_high_s16(a, b);

#define vmulla_lo(d, c, a, b) d = vmlal_s16(c, a, b);
#define vmulla_hi(d, c, a, b) d = vmlal_high_s16(c, a, b);

#define vadd(c, a, b) c = vaddq_s16(a, b);

#define vaddv(c, a) c = vaddvq_s32(a);

#define vor(c, a, b) c = vorrq_s32(a, b);

#endif