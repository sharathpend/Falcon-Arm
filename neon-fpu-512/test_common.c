#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "inner.h"
#include "macro.h"

static const uint32_t l2bound[] = {
    0, /* unused */
    101498,
    208714,
    428865,
    892039,
    1852696,
    3842630,
    7959734,
    16468416,
    34034726,
    70265242};

/* see inner.h */
int
    Zf(is_short)(
        const int16_t *s1, const int16_t *s2, const unsigned logn)
{
    /*
	 * We use the l2-norm. Code below uses only 32-bit operations to
	 * compute the square of the norm with saturation to 2^32-1 if
	 * the value exceeds 2^31-1.
	 */
    size_t n, u;
    uint32_t s, ng;

    n = (size_t)1 << logn;
    s = 0;
    ng = 0;
    for (u = 0; u < n; u++)
    {
        int32_t z;

        z = s1[u];
        s += (uint32_t)(z * z);
        ng |= s;

        z = s2[u];
        s += (uint32_t)(z * z);
        ng |= s;
    }
    s |= -(ng >> 31);
    printf("ref  ng-s: %u-%u\n", ng, s);
    return s <= l2bound[logn];
}

int
    Zf(neon_is_short)(
        const int16_t *s1, const int16_t *s2, const unsigned logn)
{
    /*
	 * We use the l2-norm. Code below uses only 32-bit operations to
	 * compute the square of the norm with saturation to 2^32-1 if
	 * the value exceeds 2^31-1.
	 */
    size_t n;
    uint32_t s, ng;

    n = (size_t)1 << logn;
    s = 0;
    ng = 0;
    int16x8x4_t z1, z2;
    int32x4x4_t s_lo, s_hi, ngx4;
    s_lo = vld4q_dup_s32(0);
    s_hi = vld4q_dup_s32(0);
    ngx4 = vld4q_dup_s32(0);
    for (size_t u = 0; u < n; u += 32)
    {
        z1 = vld1q_u16_x4(&s1[u]);
        z2 = vld1q_u16_x4(&s2[u]);

        vmulla_lo(s_lo.val[0], s_lo.val[0], z1.val[0], z1.val[0]);
        vmulla_hi(s_hi.val[0], s_hi.val[0], z1.val[0], z1.val[0]);
        vor(ngx4.val[0], ngx4.val[0], s_lo.val[0]);
        vor(ngx4.val[0], ngx4.val[0], s_hi.val[0]);
        
        vmulla_lo(s_lo.val[1], s_lo.val[1], z1.val[1], z1.val[1]);
        vmulla_hi(s_hi.val[1], s_hi.val[1], z1.val[1], z1.val[1]);
        vor(ngx4.val[1], ngx4.val[1], s_lo.val[1]);
        vor(ngx4.val[1], ngx4.val[1], s_hi.val[1]);
        
        vmulla_lo(s_lo.val[2], s_lo.val[2], z1.val[2], z1.val[2]);
        vmulla_hi(s_hi.val[2], s_hi.val[2], z1.val[2], z1.val[2]);
        vor(ngx4.val[2], ngx4.val[2], s_lo.val[2]);
        vor(ngx4.val[2], ngx4.val[2], s_hi.val[2]);

        vmulla_lo(s_lo.val[3], s_lo.val[3], z1.val[3], z1.val[3]);
        vmulla_hi(s_hi.val[3], s_hi.val[3], z1.val[3], z1.val[3]);
        vor(ngx4.val[3], ngx4.val[3], s_lo.val[3]);
        vor(ngx4.val[3], ngx4.val[3], s_hi.val[3]);

        vmulla_lo(s_lo.val[0], s_lo.val[0], z2.val[0], z2.val[0]);
        vmulla_hi(s_hi.val[0], s_hi.val[0], z2.val[0], z2.val[0]);
        vor(ngx4.val[0], ngx4.val[0], s_lo.val[0]);
        vor(ngx4.val[0], ngx4.val[0], s_hi.val[0]);

        vmulla_lo(s_lo.val[1], s_lo.val[1], z2.val[1], z2.val[1]);
        vmulla_hi(s_hi.val[1], s_hi.val[1], z2.val[1], z2.val[1]);
        vor(ngx4.val[1], ngx4.val[1], s_lo.val[1]);
        vor(ngx4.val[1], ngx4.val[1], s_hi.val[1]);

        vmulla_lo(s_lo.val[2], s_lo.val[2], z2.val[2], z2.val[2]);
        vmulla_hi(s_hi.val[2], s_hi.val[2], z2.val[2], z2.val[2]);
        vor(ngx4.val[2], ngx4.val[2], s_lo.val[2]);
        vor(ngx4.val[2], ngx4.val[2], s_hi.val[2]);

        vmulla_lo(s_lo.val[3], s_lo.val[3], z2.val[3], z2.val[3]);
        vmulla_hi(s_hi.val[3], s_hi.val[3], z2.val[3], z2.val[3]);
        vor(ngx4.val[3], ngx4.val[3], s_lo.val[3]);
        vor(ngx4.val[3], ngx4.val[3], s_hi.val[3]);
    }
    // Collapse s_lo, s_hi
    vadd(s_lo.val[0], s_lo.val[0], s_hi.val[0]);
    vadd(s_lo.val[1], s_lo.val[1], s_hi.val[1]);
    vadd(s_lo.val[2], s_lo.val[2], s_hi.val[2]);
    vadd(s_lo.val[3], s_lo.val[3], s_hi.val[3]);
    
    vor(ngx4.val[0], ngx4.val[0], s_lo.val[0]);
    vor(ngx4.val[1], ngx4.val[1], s_lo.val[1]);
    vor(ngx4.val[2], ngx4.val[2], s_lo.val[2]);
    vor(ngx4.val[3], ngx4.val[3], s_lo.val[3]);
    
    // Collapse s_lo 4 down to 2
    vadd(s_lo.val[0], s_lo.val[0], s_lo.val[2]);
    vadd(s_lo.val[1], s_lo.val[1], s_lo.val[3]);
    vor(ngx4.val[0], ngx4.val[0], s_lo.val[0]);
    vor(ngx4.val[1], ngx4.val[1], s_lo.val[1]);

    // Collapse ngx4 4 downto 2
    vor(ngx4.val[0], ngx4.val[0], ngx4.val[2]);
    vor(ngx4.val[1], ngx4.val[1], ngx4.val[3]);
    // Collapse ngx4 2 downto 1
    vor(ngx4.val[0], ngx4.val[0], ngx4.val[1]);
    // Collapse s_lo 2 downto 1
    vadd(s_lo.val[0], s_lo.val[0], s_lo.val[1]);
    vor(ngx4.val[0], ngx4.val[0], s_lo.val[0]);
    
    uint32x2_t tmp, sx2, ngx2;
    // Collapse s_lox4 down to x2
    sx2 = vadd_u32(vget_low_u32(s_lo.val[0]), vget_high_u32(s_lo.val[0]));
    // Collapse ngx4 downto x2
    ngx2 = vorr_s32(vget_low_s32(ngx4.val[0]), vget_high_s32(ngx4.val[1]));
    ngx2 = vorr_s32(ngx2, sx2);
    // Collapse s_lox2 downto x1
    s = vaddv_s32(sx2);
    // No instruction to collapse ngx2 to ngx1, so move to general purpose register
    ng |= vget_lane_u32(ngx2, 0);
    ng |= vget_lane_u32(ngx2, 1);
    ng |= s; 
    s |= -(ng >> 31);
    printf("neon ng-s: %u-%u\n", ng, s);

    return s <= l2bound[logn];
}

/* see inner.h */
int
    Zf(neon_is_short_half)(
        uint32_t sqn, const int16_t *s2, const unsigned logn)
{
    size_t n, u;
    uint32_t ng;

    n = (size_t)1 << logn;
    ng = -(sqn >> 31);
    for (u = 0; u < n; u++)
    {
        int32_t z;

        z = s2[u];
        sqn += (uint32_t)(z * z);
        ng |= sqn;
    }
    sqn |= -(ng >> 31);

    return sqn <= l2bound[logn];
}

int main()
{
}