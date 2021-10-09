// c <= addr x1
#define vload(c, addr) c = vld1q_f64(addr);
// c <= addr interleave 4
#define vload4(c, addr) c = vld4q_f64(addr);
// c <= addr x4
#define vloadx4(c, addr) c = vld1q_f64_x4(addr);
// c <= addr interleave 2
#define vload2(c, addr) c = vld2q_f64(addr);
// c <= addr x2
#define vloadx2(c, addr) c = vld1q_f64_x2(addr);
// addr <= c
#define vstorex2(addr, c) vst1q_f64_x2(addr, c);
// addr <= c
#define vstorex4(addr, c) vst1q_f64_x4(addr, c);
// c = a - b
#define vfsub(c, a, b) c = vsubq_f64(a, b);
// c = a + b
#define vfadd(c, a, b) c = vaddq_f64(a, b);
// c = a * b
#define vfmul(c, a, b) c = vmulq_f64(a, b);
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


#define transpose(a, b, t, ia, ib, it)            \
    t.val[it] = a.val[ia];                        \
    a.val[ia] = vzip1q_f64(t.val[it], b.val[ib]); \
    b.val[ib] = vzip2q_f64(t.val[it], b.val[ib]);

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
