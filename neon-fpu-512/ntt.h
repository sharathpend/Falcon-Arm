#ifndef NTT_H
#define NTT_H

#include "config.h"

void neon_fwdNTT(int16_t a[FALCON_N], const char mont);

void neon_invNTT(int16_t a[FALCON_N]);

void neon_conv_small(int16_t out[FALCON_N], const int8_t in[FALCON_N]);

void neon_div_12289(int16_t f[FALCON_N], const int16_t g[FALCON_N]);

void neon_poly_sub(int16_t h[FALCON_N], const int16_t c0[FALCON_N], const int16_t s1[FALCON_N]);

void neon_poly_unsigned(int16_t f[FALCON_N]);

uint16_t neon_compare_with_zero(int16_t f[FALCON_N]);

void neon_poly_montymul_ntt(int16_t *f, const int16_t *g);

void neon_poly_sub_barrett(int16_t *f, const int16_t *g);

int neon_big_to_smallints(int8_t G[FALCON_N], const int16_t t[FALCON_N]);

#endif
