## Disable Complex Mul

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_add |        0 |        6
| poly_add |        1 |        4
| poly_add |        2 |        4
| poly_add |        3 |        6
| poly_add |        4 |        9
| poly_add |        5 |       16
| poly_add |        6 |       31
| poly_add |        7 |       62
| poly_add |        8 |      193
| poly_add |        9 |      317
| poly_add |       10 |      603

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_sub |        0 |        6
| poly_sub |        1 |        4
| poly_sub |        2 |        4
| poly_sub |        3 |        6
| poly_sub |        4 |        9
| poly_sub |        5 |       16
| poly_sub |        6 |       31
| poly_sub |        7 |       62
| poly_sub |        8 |      193
| poly_sub |        9 |      317
| poly_sub |       10 |      643

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_neg |        0 |        5
| poly_neg |        1 |        4
| poly_neg |        2 |        4
| poly_neg |        3 |        5
| poly_neg |        4 |        9
| poly_neg |        5 |       18
| poly_neg |        6 |       36
| poly_neg |        7 |       71
| poly_neg |        8 |      204
| poly_neg |        9 |      346
| poly_neg |       10 |      563

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_adj_fft |        0 |        5
| poly_adj_fft |        1 |        3
| poly_adj_fft |        2 |        4
| poly_adj_fft |        3 |        4
| poly_adj_fft |        4 |        5
| poly_adj_fft |        5 |        8
| poly_adj_fft |        6 |       15
| poly_adj_fft |        7 |       34
| poly_adj_fft |        8 |      132
| poly_adj_fft |        9 |      141
| poly_adj_fft |       10 |      283

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_fft |        0 |        5
| poly_mul_fft |        1 |        5
| poly_mul_fft |        2 |        4
| poly_mul_fft |        3 |        6
| poly_mul_fft |        4 |       12
| poly_mul_fft |        5 |       20
| poly_mul_fft |        6 |       36
| poly_mul_fft |        7 |       70
| poly_mul_fft |        8 |      207
| poly_mul_fft |        9 |      345
| poly_mul_fft |       10 |      665

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_invnorm2_fft |        0 |        5
| poly_invnorm2_fft |        1 |        4
| poly_invnorm2_fft |        2 |        5
| poly_invnorm2_fft |        3 |        6
| poly_invnorm2_fft |        4 |       13
| poly_invnorm2_fft |        5 |       22
| poly_invnorm2_fft |        6 |       40
| poly_invnorm2_fft |        7 |       76
| poly_invnorm2_fft |        8 |      160
| poly_invnorm2_fft |        9 |      331
| poly_invnorm2_fft |       10 |      629

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_autoadj_fft |        0 |        5
| poly_mul_autoadj_fft |        1 |        5
| poly_mul_autoadj_fft |        2 |        4
| poly_mul_autoadj_fft |        3 |        7
| poly_mul_autoadj_fft |        4 |       10
| poly_mul_autoadj_fft |        5 |       18
| poly_mul_autoadj_fft |        6 |       42
| poly_mul_autoadj_fft |        7 |       84
| poly_mul_autoadj_fft |        8 |      238
| poly_mul_autoadj_fft |        9 |      400
| poly_mul_autoadj_fft |       10 |      753

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDL_fft |        0 |       11
| poly_LDL_fft |        1 |       28
| poly_LDL_fft |        2 |       23
| poly_LDL_fft |        3 |       23
| poly_LDL_fft |        4 |       29
| poly_LDL_fft |        5 |       52
| poly_LDL_fft |        6 |      100
| poly_LDL_fft |        7 |      196
| poly_LDL_fft |        8 |      518
| poly_LDL_fft |        9 |      916
| poly_LDL_fft |       10 |     1850

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDLmv_fft |        0 |       11
| poly_LDLmv_fft |        1 |       11
| poly_LDLmv_fft |        2 |       11
| poly_LDLmv_fft |        3 |       16
| poly_LDLmv_fft |        4 |       26
| poly_LDLmv_fft |        5 |       46
| poly_LDLmv_fft |        6 |       86
| poly_LDLmv_fft |        7 |      166
| poly_LDLmv_fft |        8 |      403
| poly_LDLmv_fft |        9 |      726
| poly_LDLmv_fft |       10 |     1370

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_split_fft |        0 |        5
| poly_split_fft |        1 |        3
| poly_split_fft |        2 |        5
| poly_split_fft |        3 |        7
| poly_split_fft |        4 |       18
| poly_split_fft |        5 |       32
| poly_split_fft |        6 |       60
| poly_split_fft |        7 |      116
| poly_split_fft |        8 |      228
| poly_split_fft |        9 |      453
| poly_split_fft |       10 |      915

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |       11
| poly_merge_fft |        1 |       11
| poly_merge_fft |        2 |       11
| poly_merge_fft |        3 |       11
| poly_merge_fft |        4 |       15
| poly_merge_fft |        5 |       28
| poly_merge_fft |        6 |       50
| poly_merge_fft |        7 |       94
| poly_merge_fft |        8 |      182
| poly_merge_fft |        9 |      358
| poly_merge_fft |       10 |      710



## Enable Complex MUL

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_add |        0 |        6
| poly_add |        1 |        4
| poly_add |        2 |        4
| poly_add |        3 |        6
| poly_add |        4 |        9
| poly_add |        5 |       16
| poly_add |        6 |       31
| poly_add |        7 |       62
| poly_add |        8 |      193
| poly_add |        9 |      317
| poly_add |       10 |      572

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_sub |        0 |        6
| poly_sub |        1 |        4
| poly_sub |        2 |        4
| poly_sub |        3 |        6
| poly_sub |        4 |        9
| poly_sub |        5 |       16
| poly_sub |        6 |       31
| poly_sub |        7 |       62
| poly_sub |        8 |      193
| poly_sub |        9 |      317
| poly_sub |       10 |      592

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_neg |        0 |        5
| poly_neg |        1 |        4
| poly_neg |        2 |        4
| poly_neg |        3 |        5
| poly_neg |        4 |        9
| poly_neg |        5 |       18
| poly_neg |        6 |       36
| poly_neg |        7 |       72
| poly_neg |        8 |      205
| poly_neg |        9 |      346
| poly_neg |       10 |      556

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_adj_fft |        0 |        5
| poly_adj_fft |        1 |        3
| poly_adj_fft |        2 |        4
| poly_adj_fft |        3 |        4
| poly_adj_fft |        4 |        5
| poly_adj_fft |        5 |        8
| poly_adj_fft |        6 |       15
| poly_adj_fft |        7 |       32
| poly_adj_fft |        8 |      131
| poly_adj_fft |        9 |      141
| poly_adj_fft |       10 |      283

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_fft |        0 |        5
| poly_mul_fft |        1 |        5
| poly_mul_fft |        2 |        4
| poly_mul_fft |        3 |        6
| poly_mul_fft |        4 |       12
| poly_mul_fft |        5 |       20
| poly_mul_fft |        6 |       36
| poly_mul_fft |        7 |       68
| poly_mul_fft |        8 |      207
| poly_mul_fft |        9 |      345
| poly_mul_fft |       10 |      665

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_invnorm2_fft |        0 |        5
| poly_invnorm2_fft |        1 |        4
| poly_invnorm2_fft |        2 |        5
| poly_invnorm2_fft |        3 |        6
| poly_invnorm2_fft |        4 |       13
| poly_invnorm2_fft |        5 |       22
| poly_invnorm2_fft |        6 |       40
| poly_invnorm2_fft |        7 |       76
| poly_invnorm2_fft |        8 |      160
| poly_invnorm2_fft |        9 |      331
| poly_invnorm2_fft |       10 |      629

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_mul_autoadj_fft |        0 |        5
| poly_mul_autoadj_fft |        1 |        5
| poly_mul_autoadj_fft |        2 |        4
| poly_mul_autoadj_fft |        3 |        7
| poly_mul_autoadj_fft |        4 |       10
| poly_mul_autoadj_fft |        5 |       17
| poly_mul_autoadj_fft |        6 |       42
| poly_mul_autoadj_fft |        7 |       86
| poly_mul_autoadj_fft |        8 |      233
| poly_mul_autoadj_fft |        9 |      409
| poly_mul_autoadj_fft |       10 |      737

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDL_fft |        0 |       11
| poly_LDL_fft |        1 |       23
| poly_LDL_fft |        2 |       23
| poly_LDL_fft |        3 |       23
| poly_LDL_fft |        4 |       29
| poly_LDL_fft |        5 |       52
| poly_LDL_fft |        6 |      100
| poly_LDL_fft |        7 |      196
| poly_LDL_fft |        8 |      545
| poly_LDL_fft |        9 |      928
| poly_LDL_fft |       10 |     1683

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_LDLmv_fft |        0 |       11
| poly_LDLmv_fft |        1 |       11
| poly_LDLmv_fft |        2 |       11
| poly_LDLmv_fft |        3 |       16
| poly_LDLmv_fft |        4 |       26
| poly_LDLmv_fft |        5 |       46
| poly_LDLmv_fft |        6 |       86
| poly_LDLmv_fft |        7 |      166
| poly_LDLmv_fft |        8 |      406
| poly_LDLmv_fft |        9 |      727
| poly_LDLmv_fft |       10 |     1368

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_split_fft |        0 |        5
| poly_split_fft |        1 |        3
| poly_split_fft |        2 |        5
| poly_split_fft |        3 |        7
| poly_split_fft |        4 |       18
| poly_split_fft |        5 |       32
| poly_split_fft |        6 |       60
| poly_split_fft |        7 |      116
| poly_split_fft |        8 |      228
| poly_split_fft |        9 |      453
| poly_split_fft |       10 |      905

| Function | logn | cycles |
|:-------------|----------:|-----------:|
| poly_merge_fft |        0 |       11
| poly_merge_fft |        1 |       11
| poly_merge_fft |        2 |       11
| poly_merge_fft |        3 |       11
| poly_merge_fft |        4 |       15
| poly_merge_fft |        5 |       28
| poly_merge_fft |        6 |       50
| poly_merge_fft |        7 |       94
| poly_merge_fft |        8 |      182
| poly_merge_fft |        9 |      358
| poly_merge_fft |       10 |      710