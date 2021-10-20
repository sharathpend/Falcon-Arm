#ifndef SIGN_H
#define SIGN_H

#include "fpr.h"

/*
 * Compute degree N from logarithm 'logn'.
 */
// TODO: replace by FALCON_N
#define MKN(logn)   ((size_t)1 << (logn))


typedef int (*samplerZ)(void *ctx, fpr mu, fpr sigma);

int PQCLEAN_FALCON512_NEON_sampler(void *ctx, fpr mu, fpr isigma);

int PQCLEAN_FALCON512_NEON_gaussian0_sampler(prng *p);

void PQCLEAN_FALCON512_NEON_expand_privkey(fpr *expanded_key,
                                           const int8_t *f, const int8_t *g,
                                           const int8_t *F, const int8_t *G,
                                           unsigned logn, uint8_t *tmp);

void
smallints_to_fpr(fpr *r, const int8_t *t, unsigned logn); 

#endif