#include <arm_neon.h>
#include <stdio.h>

// gcc -o test reduction_test.c -O3; ./test

#define FALCON_Q 12289
#define FALCON_QINV (-12287)

int16_t mul(int16_t a, int16_t b)
{
    int16_t c;
    c = ((int32_t)a * b) % FALCON_Q;
    return c;
}

/*
 * If `sl, sr` = 25, 26 then `v` = 5461
 * If `sl, sr` = 26, 27 then `v` = 10922
 * If `sl, sr` = 27, 28 then `v` = 21844
 * I select `v` = 5461
 */
int16_t barrett_reduce(int16_t a)
{
    int16_t t;
    const int sl = 25;
    const int sr = 26;
    const int16_t v = ((1 << sr) + FALCON_Q / 2) / FALCON_Q;

    t = ((int32_t)v * a + (1 << sl)) >> sr;
    t *= FALCON_Q;
    return a - t;
}

/*
 * If vrshr `n` = 11, then v = 5461
 * If vrshr `n` = 12, then v = 10922
 * If vrshr `n` = 13, then v = 21843, 21844
 * I select `v` = 5461, `n` = 11
 */
int16_t barrett_simd(int16_t a)
{
    int16x8_t t, t1, z;

    z = vdupq_n_s16(a);

    t = vqdmulhq_n_s16(z, 5461);
    t = vrshrq_n_s16(t, 11);
    z = vmlsq_n_s16(z, t, FALCON_Q);

    return z[0];
}

/* 
 * Output in [-Q/2, Q/2]
 * a in [-R, R]
 * c = a % Q => c in [-Q/2, Q/2]
 */
int test_barret_red()
{
    int16_t gold, test;
    int16_t min = INT16_MAX, max = INT16_MIN;
    printf("test_barrett_reduction: ");
    for (int16_t i = INT16_MIN; i < INT16_MAX; i++)
    {
        gold = barrett_reduce(i);
        test = barrett_simd(i);

        if (test < min)
            min = test;
        if (max < test)
            max = test;

        if (gold != test)
        {
            printf("[%d] [%d]\n", gold, test);
        }
    }

    printf("OK\n");
    printf("min, max = %d, %d\n", min, max);
    return 0;
}

/*
def compute_doubling_16(a):
    Q = 12289
    QINV = 53249
    root = a * pow(2, 16) % Q
    twisted_root = (pow(2, 16) - root * QINV) % pow(2, 16)
    return root, twisted_root
*/
void montgomery_doubling_root(int16_t b, int16_t *broot, int16_t *btwisted)
{
    int32_t root, twisted_root;

    root = (int32_t)b;
    root = (root << 16) % FALCON_Q;
    twisted_root = (root * FALCON_QINV) % (1 << 16);

    *broot = root;
    *btwisted = twisted_root;
}

int16_t montgomery_doubling(int16_t a, int16_t b)
{
    int16x8_t neon_a, neon_z, neon_t;
    int16_t broot, btwisted;

    montgomery_doubling_root(b, &broot, &btwisted);
    // printf("broot, btwisted: %d %d\n", broot, btwisted);

    neon_a = vdupq_n_s16(a);

    neon_z = vqdmulhq_n_s16(neon_a, broot);
    neon_a = vmulq_n_s16(neon_a, btwisted);
    neon_a = vqdmulhq_n_s16(neon_a, FALCON_Q);
    neon_z = vhsubq_s16(neon_z, neon_a);

    return neon_z[0];
}

/*
 * Doubling work full range [-R, R]
 * a in [-R, R]
 * b in [-R, R]
 * c = a*b => c in [-R, R]
 */
int test_montgomery_doubling()
{
    printf("test_montgomery_doubling: ");
    int16_t gold, test;
    int16_t min = INT16_MAX, max = INT16_MIN;
    for (int16_t a = INT16_MIN; a < INT16_MAX; a++)
    {
        for (int16_t b = INT16_MIN; b < INT16_MAX; b++)
        {
            gold = mul(a, b);
            test = montgomery_doubling(a, b);

            if (test < min)
                min = test;
            if (max < test)
                max = test;

            gold = (gold + FALCON_Q) % FALCON_Q;
            test = (test + FALCON_Q) % FALCON_Q;

            if ((gold != test) && (gold != test + FALCON_Q))
            {
                printf("\n");
                printf("Error %d * %d: %d != %d\n", a, b, gold, test);
                printf("Error %d * %d: %d != %d\n", a, b, gold, test + FALCON_Q);
                return 1;
            }
        }
    }
    printf("OK\n");
    printf("min, max = %d, %d\n", min, max);

    return 0;
}

/*
def compute(a):
    Q = 12289
    QINV = 53249
    root = a * pow(2, 15) % Q
    if root % 2 == 0:
        root += Q

    twisted_root = (pow(2, 16) - root * QINV) % pow(2, 16)

    return root, twisted_root
*/
void montgomery_rounding_root(int16_t b, int16_t *broot, int16_t *btwisted)
{
    int32_t root, twisted_root;

    root = (int32_t)b;
    root = (root << 15) % FALCON_Q;
    if ((root & 1) == 0)
    {
        root += FALCON_Q;
    }
    twisted_root = (-root * FALCON_QINV) & 0xFFFF;

    *broot = root;
    *btwisted = twisted_root;
}

int16_t montgomery_rounding(int16_t a, int16_t b)
{
    int16x8_t neon_a, neon_z, neon_t;
    int16_t broot, btwisted;

    montgomery_rounding_root(b, &broot, &btwisted);
    // if ( (b == 12265) || (b == 12277) )
    // printf("broot, btwisted: %d %d\n", broot, btwisted);

    neon_a = vdupq_n_s16(a);

    neon_z = vqrdmulhq_n_s16(neon_a, broot);
    neon_a = vmulq_n_s16(neon_a, btwisted);
    neon_z = vqrdmlahq_s16(neon_z, neon_a, vdupq_n_s16(FALCON_Q));

    return neon_z[0];
}

/*
 * Rounding work range 
 * a in [-R/2 + 1, R/2]
 * b in [-Q/2, Q/2]
 * c = a * b => c in [-24568, 24568]
 */
int test_montgomery_rounding()
{
    printf("test_montgomery_rounding: ");

    int16_t gold, test;
    int16_t min = INT16_MAX, max = INT16_MIN;
    for (int16_t a = INT16_MIN / 2 + 1; a < INT16_MAX / 2; a++)
    {
        for (int16_t b = -FALCON_Q/2; b < FALCON_Q/2; b++)
        {
            gold = mul(a, b);
            test = montgomery_rounding(a, b);

            if (test < min)
                min = test;
            if (max < test)
                max = test;

            gold = (gold + FALCON_Q) % FALCON_Q;
            test = (test + FALCON_Q) % FALCON_Q;

            if ((gold != test) && (gold != test + FALCON_Q))
            {
                printf("\n");
                printf("Error %d * %d: %d != %d\n", a, b, gold, test);
                printf("Error %d * %d: %d != %d\n", a, b, gold, test + FALCON_Q);
                return 1;
            }
        }
    }
    printf("OK\n");

    printf("min, max = %d, %d\n", min, max);
    return 0;
}

void barrett_rounding_root(int16_t b, int16_t *broot)
{
    int32_t root;
    root = b;
    root = ((int32_t)(root << 16) / FALCON_Q) / 2;

    *broot = root;
}

int16_t barrett_mul(int16_t a, int16_t b)
{
    int16x8_t neon_a, neon_z, neon_t;

    int16_t broot;
    barrett_rounding_root(b, &broot);

    neon_a = vdupq_n_s16(a);

    neon_z = vmulq_n_s16(neon_a, b);
    neon_t = vqrdmulhq_n_s16(neon_a, broot);
    // printf("neon_t = %d\n", neon_t[0]);
    neon_z = vmlsq_n_s16(neon_z, neon_t, FALCON_Q);

    return neon_z[0];
}

/* 
 * Output in range [-18431, 18430]
 * a in [-R, R]
 * b in [-Q/2, Q/2]
 * c = a * b => c in [-18431, 18430]
 */
int test_barrett_mul()
{
    printf("test_barrett_mul: ");

    int16_t gold, test;
    int16_t min = INT16_MAX, max = INT16_MIN;

    for (int16_t a = INT16_MIN; a < INT16_MAX; a++)
    {
        for (int16_t b = -FALCON_Q/2; b < FALCON_Q; b++)
        {
            gold = mul(a, b);
            test = barrett_mul(a, b);

            if (test < min)
                min = test;
            if (max < test)
                max = test;

            if ((gold != test) && (gold != test + FALCON_Q) && (gold != test - FALCON_Q))
            {
                printf("\n");
                printf("Error %d * %d: %d != %d\n", b, a, gold, test);
                printf("Error %d * %d: %d != %d\n", b, a, gold, test + FALCON_Q);
                return 1;
            }
        }
    }
    printf("OK\n");

    printf("min, max = %d, %d\n", min, max);
    return 0;
}

int main()
{

    int ret = 0;

    ret |= test_montgomery_rounding();
    ret |= test_montgomery_doubling();
    ret |= test_barret_red();
    ret |= test_barrett_mul();

    if (ret)
        return 1;

    return 0;
}

/*
❯ gcc -o reduction_test reduction_test.c -O3; ./reduction_test 
test_montgomery_rounding: OK
min, max = -24568, 24568
test_montgomery_doubling: OK
min, max = -12285, 12288
test_barrett_reduction: OK
min, max = -6145, 6145
test_barrett_mul: OK
min, max = -18431, 18430
 */