# Fast Compressed FFT versus reference FFT code

Iterations: 50,000

test_with_ref_FFF: -O0
Compare my FFT versus reference FFT code

| Function (logn) | Fast FFT | Ref FFT |
|:-------------|----------:|-----------:|
| FFT 2 |      104 |      445
| FFT 3 |      160 |     1265
| FFT 4 |      259 |     3619
| FFT 5 |      451 |     9194
| FFT 6 |      867 |    21943
| FFT 7 |     1756 |    50817
| FFT 8 |     3812 |   116311
| FFT 9 |     8366 |   262861
| FFT 10 |    18350 |   587092


test_with_ref_FFF: -O1
Compare my FFT versus reference FFT code

| Function (logn) | Fast FFT | Ref FFT |
|:-------------|----------:|-----------:|
| FFT 2 |      102 |      214
| FFT 3 |      159 |      362
| FFT 4 |      257 |      774
| FFT 5 |      451 |     1865
| FFT 6 |      868 |     4231
| FFT 7 |     1757 |     9559
| FFT 8 |     3812 |    21368
| FFT 9 |     8368 |    47487
| FFT 10 |    18350 |   105061


test_with_ref_FFF: -O2
Compare my FFT versus reference FFT code

| Function (logn) | Fast FFT | Ref FFT |
|:-------------|----------:|-----------:|
| FFT 2 |      100 |      126
| FFT 3 |      159 |      193
| FFT 4 |      259 |      254
| FFT 5 |      451 |      412
| FFT 6 |      868 |      754
| FFT 7 |     1757 |     1543
| FFT 8 |     3814 |     4274
| FFT 9 |     8369 |     8567
| FFT 10 |    18352 |    17820

test_with_ref_FFF: -O3
Compare my FFT versus reference FFT code

| Function (logn) | Fast FFT | Ref FFT |
|:-------------|----------:|-----------:|
| FFT 2 |      102 |      123
| FFT 3 |      159 |      182
| FFT 4 |      258 |      250
| FFT 5 |      451 |      396
| FFT 6 |      868 |      723
| FFT 7 |     1756 |     1489
| FFT 8 |     3814 |     4135
| FFT 9 |     8369 |     8329
| FFT 10 |    18352 |    17370