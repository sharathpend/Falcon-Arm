#ifndef POLY_H
#define POLY_H

#include "inner.h"
#include "config.h"

void ZfN(poly_ntt)(int16_t a[FALCON_N], const char mont);

void ZfN(poly_invntt)(int16_t a[FALCON_N]);

void ZfN(poly_smallints_to_bigints)(int16_t out[FALCON_N], const int8_t in[FALCON_N]);

void ZfN(poly_div_12289)(int16_t f[FALCON_N], const int16_t g[FALCON_N]);

void ZfN(poly_convert_to_unsigned)(int16_t f[FALCON_N]);

uint16_t ZfN(poly_compare_with_zero)(int16_t f[FALCON_N]);

void ZfN(poly_montmul_ntt)(int16_t f[FALCON_N], const int16_t g[FALCON_N]);

void ZfN(poly_sub_barrett)(int16_t f[FALCON_N], const int16_t g[FALCON_N], const int16_t s[FALCON_N]);

int ZfN(bigints_to_smallints)(int8_t G[FALCON_N], const int16_t t[FALCON_N]);

int ZfN(poly_check_bound_int8)(const int8_t t[FALCON_N],
                               const int8_t low, const int8_t high);

int ZfN(poly_check_bound_int16)(const int16_t t[FALCON_N],
                                const int16_t low, const int16_t high);

#endif
