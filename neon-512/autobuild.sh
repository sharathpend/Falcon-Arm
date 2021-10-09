#!/bin/bash
# set -e 
# gcc -o neon-fft fips202.c fpr.c  inner.c util.c fft.c neon_fft.c -O3 -Wall
# ./neon-fft

# set -e 
while inotifywait -e close_write fft.c; do 
    gcc -o neon-fft fips202.c fpr.c  inner.c util.c fft.c neon_fft.c -O3 -g3 -Wall
    # ./neon-fft | md5sum - 
    ./neon-fft
done