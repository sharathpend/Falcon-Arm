#include "ntt_consts.h"
#define PADDING 0

#if FALCON_LOGN == 9
// ❯ python table_generate_512.py >>  ntt_consts9.c
#include "ntt_consts9.c"

#elif FALCON_LOGN == 10
// ❯ python table_generate_1024.py >>  ntt_consts10.c
#include "ntt_consts10.c"

#else
#error "Only support falcon_logn = 9,10"

#endif
