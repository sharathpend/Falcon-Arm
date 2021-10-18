#!/bin/bash
# set -e 
# gcc -o neon-fft fips202.c fpr.c fft_split_merge.c inner.c util.c fft.c neon_fft.c -O3 -g3 -Wall
# ./neon-fft
# ./neon-fft | md5sum - 

set -e 
while inotifywait -e close_write fft.c neon_fft.c fft_split_merge.c; do 
    gcc -o neon-fft fips202.c fpr.c fpr_half.c fft_split_merge.c inner.c util.c fft.c neon_fft.c -O3 -g3 -Wall
    ./neon-fft
    ./neon-fft | md5sum - 
done