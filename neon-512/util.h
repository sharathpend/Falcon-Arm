#ifndef UTIL_H
#define UTIL_H

#include "inner.h"

void print_array(fpr *a, int length, const char *string, int print_float);
double fRand(double fMin, double fMax);
int compare(fpr *gold, fpr *test, int bound, const char *string);

#endif
