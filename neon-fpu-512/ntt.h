#ifndef NTT_H
#define NTT_H

/*
 * Constants for NTT.
 *
 *   n = 2^logn  (2 <= n <= 1024)
 *   phi = X^n + 1
 *   q = 12289
 *   q0i = -1/q mod 2^16
 *   R = 2^16 mod q
 *   R2 = 2^32 mod q
 */

#define Q 12289
#define Q0I 12287
#define R 4091
#define R2 10952

void mq_NTT(uint16_t *a, unsigned logn);

void mq_iNTT(uint16_t *a, unsigned logn);

extern inline uint32_t
mq_conv_small(int x);

extern inline uint32_t
mq_div_12289(uint32_t x, uint32_t y);

extern inline uint32_t
mq_sub(uint32_t x, uint32_t y);

void mq_poly_tomonty(uint16_t *f, unsigned logn);

void mq_poly_montymul_ntt(uint16_t *f, const uint16_t *g, unsigned logn);

void mq_poly_sub(uint16_t *f, const uint16_t *g, unsigned logn);

#endif
