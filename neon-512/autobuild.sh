#!/bin/bash
# set -e 
# gcc -o neon-fft fips202.c fpr.c fft_split_merge.c inner.c util.c fft.c neon_fft.c -O3 -g3 -Wall
# ./neon-fft
# ./neon-fft | md5sum - 

# set -e 
# while inotifywait -e close_write fft.c neon_fft.c fft_split_merge.c; do 
#     gcc -o neon-fft fips202.c fpr.c fpr_half.c fft_split_merge.c inner.c util.c fft.c neon_fft.c -O3 -g3 -Wall
#     ./neon-fft
#     ./neon-fft | md5sum - 
# done

# set -e 
# while inotifywait -e close_write macro.h test_fpr.c vfpr.h; do 
#     gcc -o test_fpr util.c test_fpr.c -O3 -g3 -Wall -Wextra -Wpedantic;
#     ./test_fpr
# done

while inotifywait -e close_write macro.h sampler.c test_log2.c vfpr.h; do 
    # gcc -o test_log2 util.c test_log2.c -O3 -g3 -Wall -Wextra -Wpedantic;
    gcc -o test_log2 fips202.c util.c rng.c sampler.c test_log2.c -O3 -g3
    ./test_log2
done 
