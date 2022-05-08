# REF A72

| FFT | Foward FFT | Inverse FFT
|:-------------|----------:|-----------:|
| FFT 0 |       14 |       19
| FFT 1 |       14 |       48
| FFT 2 |       38 |       50
| FFT 3 |      109 |      150
| FFT 4 |      261 |      294
| FFT 5 |      588 |      654
| FFT 6 |     1299 |     1438
| FFT 7 |     2895 |     3176
| FFT 8 |     6269 |     6946
| FFT 9 |    13520 |    15053
| FFT 10 |    29147 |    32903

| NTT | Foward NTT | Inverse NTT
|:-------------|----------:|-----------:|
| NTT 9 |    23024 |    21946
| NTT 10 |    49338 |    46481

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_add |        0 |        9
| poly_add |        1 |       13
| poly_add |        2 |       19
| poly_add |        3 |       32
| poly_add |        4 |       56
| poly_add |        5 |      103
| poly_add |        6 |      214
| poly_add |        7 |      407
| poly_add |        8 |      792
| poly_add |        9 |     1558
| poly_add |       10 |     3109

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_sub |        0 |        9
| poly_sub |        1 |       14
| poly_sub |        2 |       21
| poly_sub |        3 |       33
| poly_sub |        4 |       56
| poly_sub |        5 |      104
| poly_sub |        6 |      217
| poly_sub |        7 |      409
| poly_sub |        8 |      791
| poly_sub |        9 |     1559
| poly_sub |       10 |     3104

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_neg |        0 |        9
| poly_neg |        1 |       26
| poly_neg |        2 |       17
| poly_neg |        3 |       25
| poly_neg |        4 |       41
| poly_neg |        5 |       78
| poly_neg |        6 |      172
| poly_neg |        7 |      321
| poly_neg |        8 |      620
| poly_neg |        9 |     1218
| poly_neg |       10 |     2412

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_adj_fft |        0 |        9
| poly_adj_fft |        1 |        9
| poly_adj_fft |        2 |       28
| poly_adj_fft |        3 |       11
| poly_adj_fft |        4 |       37
| poly_adj_fft |        5 |       26
| poly_adj_fft |        6 |       47
| poly_adj_fft |        7 |       87
| poly_adj_fft |        8 |      167
| poly_adj_fft |        9 |      346
| poly_adj_fft |       10 |      666

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_fft |        0 |        9
| poly_mul_fft |        1 |       12
| poly_mul_fft |        2 |       16
| poly_mul_fft |        3 |       30
| poly_mul_fft |        4 |       52
| poly_mul_fft |        5 |       92
| poly_mul_fft |        6 |      172
| poly_mul_fft |        7 |      351
| poly_mul_fft |        8 |      671
| poly_mul_fft |        9 |     1311
| poly_mul_fft |       10 |     2766

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_invnorm2_fft |        0 |        9
| poly_invnorm2_fft |        1 |       13
| poly_invnorm2_fft |        2 |       24
| poly_invnorm2_fft |        3 |       53
| poly_invnorm2_fft |        4 |      106
| poly_invnorm2_fft |        5 |      238
| poly_invnorm2_fft |        6 |      494
| poly_invnorm2_fft |        7 |     1013
| poly_invnorm2_fft |        8 |     2037
| poly_invnorm2_fft |        9 |     4111
| poly_invnorm2_fft |       10 |     8213

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_autoadj_fft |        0 |        9
| poly_mul_autoadj_fft |        1 |       10
| poly_mul_autoadj_fft |        2 |       13
| poly_mul_autoadj_fft |        3 |       21
| poly_mul_autoadj_fft |        4 |       35
| poly_mul_autoadj_fft |        5 |       57
| poly_mul_autoadj_fft |        6 |      101
| poly_mul_autoadj_fft |        7 |      207
| poly_mul_autoadj_fft |        8 |      383
| poly_mul_autoadj_fft |        9 |      735
| poly_mul_autoadj_fft |       10 |     1439

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDL_fft |        0 |        9
| poly_LDL_fft |        1 |       38
| poly_LDL_fft |        2 |       51
| poly_LDL_fft |        3 |       94
| poly_LDL_fft |        4 |      179
| poly_LDL_fft |        5 |      349
| poly_LDL_fft |        6 |      689
| poly_LDL_fft |        7 |     1426
| poly_LDL_fft |        8 |     2845
| poly_LDL_fft |        9 |     5474
| poly_LDL_fft |       10 |    12346

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDLmv_fft |        0 |        9
| poly_LDLmv_fft |        1 |       40
| poly_LDLmv_fft |        2 |       56
| poly_LDLmv_fft |        3 |      100
| poly_LDLmv_fft |        4 |      200
| poly_LDLmv_fft |        5 |      387
| poly_LDLmv_fft |        6 |      732
| poly_LDLmv_fft |        7 |     1440
| poly_LDLmv_fft |        8 |     2857
| poly_LDLmv_fft |        9 |     5748
| poly_LDLmv_fft |       10 |    11868

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_split_fft |        0 |        9
| poly_split_fft |        1 |        9
| poly_split_fft |        2 |       25
| poly_split_fft |        3 |       36
| poly_split_fft |        4 |       58
| poly_split_fft |        5 |       99
| poly_split_fft |        6 |      183
| poly_split_fft |        7 |      465
| poly_split_fft |        8 |      878
| poly_split_fft |        9 |     1757
| poly_split_fft |       10 |     3575

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |        9
| poly_merge_fft |        1 |        9
| poly_merge_fft |        2 |       25
| poly_merge_fft |        3 |       77
| poly_merge_fft |        4 |       83
| poly_merge_fft |        5 |      123
| poly_merge_fft |        6 |      203
| poly_merge_fft |        7 |      387
| poly_merge_fft |        8 |      704
| poly_merge_fft |        9 |     1369
| poly_merge_fft |       10 |     3275


| degree | kg(ms) |  ek(us) |  sd(us) | sdc(us) |  st(us) | stc(us) |  vv(us) | vvc(us)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512: |   19.51 |  275.68 |  865.79 |  911.64 |  514.76 |  560.33 |   72.57 |  117.50
| 1024: |   59.13 |  580.92 | 1779.01 | 1866.69 | 1036.55 | 1122.94 |  152.94 |  235.49

| degree |  kg(kc)   | ek(kc) |  sd(kc) | sdc(kc) |  st(kc) | stc(kc) |  vv(kc) | vvc(kc)
| ---- | ------ | -- | -- | --- |--- |--- |---| --- |
|  512: |  33602.87 |  494.99 | 1553.44 | 1636.31 |  918.57 | 1001.67 |  127.81 |  211.21
| 1024: |  93555.89 | 1044.08 | 3192.95 | 3351.56 | 1857.49 | 2015.62 |  272.07 |  423.32

