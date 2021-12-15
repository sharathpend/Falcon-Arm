#ifndef ORIG_NTT_H
#define ORIG_NTT_H

#include <arm_neon.h>
#include <stddef.h>

#define Q 12289
#define Q0I 12287
#define R 4091
#define R2 10952

void original_mq_NTT(uint16_t *a, unsigned logn);

void original_mq_iNTT(uint16_t *a, unsigned logn);

#endif
