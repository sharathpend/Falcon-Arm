#include <arm_neon.h>
#include <stdio.h>

// gcc -o test reduction_test.c -O3; ./test

#define FALCON_Q 12289
#define FALCON_QINV (-12287)
#define FALCON_MONT 4091

int16_t mul(int16_t a, int16_t b)
{
    int16_t c;
    c = ((int32_t)a * b) % FALCON_Q;
    return c;
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
 */
int test_montgomery_doubling()
{
    printf("test_montgomery_doubling: ");
    int16_t gold, test;
    for (int16_t a = INT16_MIN; a < INT16_MAX; a++)
    {
        for (int16_t b = INT16_MIN; b < INT16_MAX; b++)
        {
            gold = mul(a, b);
            test = montgomery_doubling(a, b);

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
    twisted_root = ((1 << 16) - root * FALCON_QINV) % (1 << 16);

    *broot = root;
    *btwisted = twisted_root;
}

int16_t montgomery_rounding(int16_t a, int16_t b)
{
    int16x8_t neon_a, neon_z, neon_t;
    int16_t broot, btwisted;

    montgomery_rounding_root(b, &broot, &btwisted);
    // printf("broot, btwisted: %d %d\n", broot, btwisted);

    neon_a = vdupq_n_s16(a);

    neon_z = vqrdmulhq_n_s16(neon_a, broot);
    neon_a = vmulq_n_s16(neon_a, btwisted);
    neon_z = vqrdmlahq_s16(neon_z, neon_a, vdupq_n_s16(FALCON_Q));

    return neon_z[0];
}

/* 
 * Rounding work range [-R/2 + 1, R/2]
 */
int test_montgomery_rounding()
{
    printf("test_montgomery_rounding: ");

    int16_t gold, test;
    for (int16_t a = INT16_MIN / 2 + 1; a < INT16_MAX / 2; a++)
    {
        for (int16_t b = INT16_MIN / 2 + 1; b < INT16_MAX / 2; b++)
        {
            gold = mul(a, b);
            test = montgomery_rounding(a, b);

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

    return 0;
}

int main()
{

    int ret = 0;

    ret |= test_montgomery_rounding();
    ret |= test_montgomery_doubling();

    if (ret)
        return 1;

    return 0;
}

/* 
❯ gcc -o test reduction_test.c -O3; ./test
test_montgomery_rounding: OK
test_montgomery_doubling: OK
 */