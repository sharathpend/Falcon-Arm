#include "ntt_consts.h"

#if FALCON_LOGN == 9

#include "ntt_consts9.c"

#elif FALCON_LOGN == 10
#include "ntt_consts10.c"

#elif FALCON_LOGN == 8
#include "ntt_consts8.c"

#else
#error "Only support falcon_logn = 8,9,10"

#endif
