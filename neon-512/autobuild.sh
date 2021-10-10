#!/bin/bash
set -e 
gcc -o neon-fft fips202.c fpr.c  inner.c util.c fft.c neon_fft.c -O0 -g3 -Wall
./neon-fft
./neon-fft | md5sum - 

# set -e 
# while inotifywait -e close_write fft.c; do 
#     gcc -o neon-fft fips202.c fpr.c  inner.c util.c fft.c neon_fft.c -O0 -g3 -Wall
#     ./neon-fft
#     ./neon-fft | md5sum - 
# done