#ifndef NTT_CONSTS
#define NTT_CONSTS

#include <stdint.h>
#include "config.h"


/*
 * Table for NTT, binary case:
 * where g = 7 (it is a 2048-th primitive root of 1 modulo q)
 */
extern const int16_t ntt_mont[];
extern const int16_t ntt_qinv_mont[];


/*
 * Table for inverse NTT
 * Since g = 7, 1/g = 8778 mod 12289.
 */

extern const int16_t invntt_mont[];
extern const int16_t invntt_qinv_mont[];

#endif 
