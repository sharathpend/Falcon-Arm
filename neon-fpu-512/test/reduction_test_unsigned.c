#include "barrett_test_unsigned.c"

// gcc -o test reduction_test.c -O3; ./test

#define ASSIGN(x, y, z, w) {x = y; z = w;}

int test_barrett_red(int shift, int k, int i, int j, int m)
{
    switch (shift)
    {
    case 11:
        search(k, barrett_simd_11, "barrett_simd_11", i, j, m, 1);
        break;
    
    case 12:
        search(k, barrett_simd_12, "barrett_simd_12", i, j, m, 1);
        break;

    case 13:
        search(k, barrett_simd_13, "barrett_simd_13", i, j, m, 1);
        break;
    
    case 14:
        search(k, barrett_simd_14, "barrett_simd_14", i, j, m, 1);
        break;
    
    case 15:
        search(k, barrett_simd_15, "barrett_simd_15", i, j, m, 1);
        break;
    }
    return 0;
}

int search_barrett_red()
{
    int c11 = 0, c12 = 0, c13 = 0, c14 = 0, c15 = 0;
    int s11 = 0, s12 = 0, s13 = 0, s14 = 0, s15 = 0;
    int k11 = 0, k12 = 0, k13 = 0, k14 = 0, k15 = 0;

    for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
    for (int m = 0; m < 2; m++)
    {
        for (uint16_t k = 0; k < UINT16_MAX; k++)
        {
            c11 = search(k, barrett_simd_11, "barrett_simd_11", i, j, m, 0);
            c12 = search(k, barrett_simd_12, "barrett_simd_12", i, j, m, 0);
            c13 = search(k, barrett_simd_13, "barrett_simd_13", i, j, m, 0);
            c14 = search(k, barrett_simd_14, "barrett_simd_14", i, j, m, 0);
            c15 = search(k, barrett_simd_15, "barrett_simd_15", i, j, m, 0);

            if (s11 <= c11) ASSIGN(s11, c11, k11, k);
            if (s12 <= c12) ASSIGN(s12, c12, k12, k);
            if (s13 <= c13) ASSIGN(s13, c13, k13, k);
            if (s14 <= c14) ASSIGN(s14, c14, k14, k);
            if (s15 <= c15) ASSIGN(s15, c15, k15, k);

            if (c14 > 32768)
            {
                printf("k = %d, count = %d\n", k, c14);
            }
        }
        printf("ROUNDING, SHIFT_ROUNDING, SHIFT_SIGNED = %d, %d, %d:\n", i, j, m);
        printf("11: %d [%d]\n", s11, k11);
        printf("12: %d [%d]\n", s12, k12);
        printf("13: %d [%d]\n", s13, k13);
        printf("14: %d [%d]\n", s14, k14);
        printf("15: %d [%d]\n", s15, k15);
        s11 = 0; k11 = 0;
        s12 = 0; k12 = 0;
        s13 = 0; k13 = 0;
        s14 = 0; k14 = 0;
        s15 = 0; k15 = 0;
    }


    return 0;
}


int main()
{

    int ret = 0;

    ret |= search_barrett_red();
    // ret |= test_barrett_red(14, 21844, 1, 1, 0);

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