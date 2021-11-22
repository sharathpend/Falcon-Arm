#include "ntt_consts.h"

#if FALCON_LOGN == 9
// python gen_ntt.py 9 > ntt_consts9.c
#include "ntt_consts9.c"

#elif FALCON_LOGN == 10
// python gen_ntt.py 10 > ntt_consts10.c
#include "ntt_consts10.c"

#elif FALCON_LOGN == 8
// python gen_ntt.py 8 > ntt_consts8.c
#include "ntt_consts8.c"

#else
#error "Only support falcon_logn = 8,9,10"

#endif
