#ifndef UTIL_H
#define UTIL_H

#define poly_small_to_fp(r, t, logn) smallints_to_fpr(r, t, logn)

void smallints_to_fpr(fpr *r, const int8_t *t, unsigned logn);

void print_farray(fpr *r, unsigned logn);

void print_iarray(int8_t *a);

#endif
