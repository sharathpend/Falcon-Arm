# NEON A72

| FFT | Foward FFT | Inverse FFT
|:-------------|----------:|-----------:|
| FFT 0 |       10 |       10
| FFT 1 |       10 |       10
| FFT 2 |       13 |       15
| FFT 3 |       41 |       44
| FFT 4 |       85 |       90
| FFT 5 |      299 |      259
| FFT 6 |      784 |      703
| FFT 7 |     1689 |     1510
| FFT 8 |     4048 |     3699
| FFT 9 |     8583 |     7737
| FFT 10 |    20140 |    18568

| NTT | Foward NTT | Inverse NTT
|:-------------|----------:|-----------:|
| NTT 9 |     4052 |     4268
| NTT 10 |     8216 |     9091

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_add |        0 |       18
| poly_add |        1 |        9
| poly_add |        2 |        9
| poly_add |        3 |       18
| poly_add |        4 |       33
| poly_add |        5 |       63
| poly_add |        6 |      123
| poly_add |        7 |      243
| poly_add |        8 |      483
| poly_add |        9 |     1050
| poly_add |       10 |     2314

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_sub |        0 |       18
| poly_sub |        1 |        9
| poly_sub |        2 |        9
| poly_sub |        3 |       18
| poly_sub |        4 |       33
| poly_sub |        5 |       63
| poly_sub |        6 |      123
| poly_sub |        7 |      243
| poly_sub |        8 |      483
| poly_sub |        9 |      993
| poly_sub |       10 |     2316

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_neg |        0 |       14
| poly_neg |        1 |        9
| poly_neg |        2 |        9
| poly_neg |        3 |       14
| poly_neg |        4 |       25
| poly_neg |        5 |       47
| poly_neg |        6 |       91
| poly_neg |        7 |      185
| poly_neg |        8 |      377
| poly_neg |        9 |      730
| poly_neg |       10 |     1434

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_adj_fft |        0 |       16
| poly_adj_fft |        1 |        9
| poly_adj_fft |        2 |        9
| poly_adj_fft |        3 |        9
| poly_adj_fft |        4 |       16
| poly_adj_fft |        5 |       26
| poly_adj_fft |        6 |       46
| poly_adj_fft |        7 |       90
| poly_adj_fft |        8 |      186
| poly_adj_fft |        9 |      378
| poly_adj_fft |       10 |      784

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_fft |        0 |        9
| poly_mul_fft |        1 |        9
| poly_mul_fft |        2 |       10
| poly_mul_fft |        3 |       20
| poly_mul_fft |        4 |       52
| poly_mul_fft |        5 |       94
| poly_mul_fft |        6 |      179
| poly_mul_fft |        7 |      349
| poly_mul_fft |        8 |      689
| poly_mul_fft |        9 |     1395
| poly_mul_fft |       10 |     3783

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_invnorm2_fft |        0 |        9
| poly_invnorm2_fft |        1 |        9
| poly_invnorm2_fft |        2 |       18
| poly_invnorm2_fft |        3 |       43
| poly_invnorm2_fft |        4 |      141
| poly_invnorm2_fft |        5 |      289
| poly_invnorm2_fft |        6 |      585
| poly_invnorm2_fft |        7 |     1177
| poly_invnorm2_fft |        8 |     2361
| poly_invnorm2_fft |        9 |     4729
| poly_invnorm2_fft |       10 |     9465

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_autoadj_fft |        0 |        9
| poly_mul_autoadj_fft |        1 |        9
| poly_mul_autoadj_fft |        2 |        9
| poly_mul_autoadj_fft |        3 |       10
| poly_mul_autoadj_fft |        4 |       30
| poly_mul_autoadj_fft |        5 |       55
| poly_mul_autoadj_fft |        6 |      105
| poly_mul_autoadj_fft |        7 |      205
| poly_mul_autoadj_fft |        8 |      405
| poly_mul_autoadj_fft |        9 |      805
| poly_mul_autoadj_fft |       10 |     1818

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDL_fft |        0 |        9
| poly_LDL_fft |        1 |       24
| poly_LDL_fft |        2 |       34
| poly_LDL_fft |        3 |       80
| poly_LDL_fft |        4 |      231
| poly_LDL_fft |        5 |      466
| poly_LDL_fft |        6 |      936
| poly_LDL_fft |        7 |     1876
| poly_LDL_fft |        8 |     3756
| poly_LDL_fft |        9 |     7537
| poly_LDL_fft |       10 |    15609

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDLmv_fft |        0 |        9
| poly_LDLmv_fft |        1 |       25
| poly_LDLmv_fft |        2 |       35
| poly_LDLmv_fft |        3 |       81
| poly_LDLmv_fft |        4 |      234
| poly_LDLmv_fft |        5 |      469
| poly_LDLmv_fft |        6 |      936
| poly_LDLmv_fft |        7 |     1875
| poly_LDLmv_fft |        8 |     3748
| poly_LDLmv_fft |        9 |     7495
| poly_LDLmv_fft |       10 |    15019

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_split_fft |        0 |        9
| poly_split_fft |        1 |        9
| poly_split_fft |        2 |       13
| poly_split_fft |        3 |       24
| poly_split_fft |        4 |       55
| poly_split_fft |        5 |       98
| poly_split_fft |        6 |      183
| poly_split_fft |        7 |      352
| poly_split_fft |        8 |      691
| poly_split_fft |        9 |     1368
| poly_split_fft |       10 |     3338

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |       10
| poly_merge_fft |        1 |        9
| poly_merge_fft |        2 |       12
| poly_merge_fft |        3 |       17
| poly_merge_fft |        4 |       43
| poly_merge_fft |        5 |       83
| poly_merge_fft |        6 |      154
| poly_merge_fft |        7 |      296
| poly_merge_fft |        8 |      580
| poly_merge_fft |        9 |     1148
| poly_merge_fft |       10 |     2930


degree  kg(ms)   ek(us)   sd(us)  sdc(us)   st(us)  stc(us)   vv(us)  vvc(us)
 512:    19.03   125.43   589.35   626.47   416.52   453.67    37.13    77.12
1024:    55.40   272.46  1219.28  1292.99   840.34   913.47    81.43   151.11


degree  kg(kc)   ek(kc)   sd(kc)  sdc(kc)   st(kc)  stc(kc)   vv(kc)  vvc(kc)
 512: 31417.68   224.88  1061.84  1129.58   749.90   816.24    68.67   138.64
1024: 88141.58   489.09  2191.13  2320.12  1510.92  1640.51   144.68   271.60