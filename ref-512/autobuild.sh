#!/bin/bash
# set -e 
# gcc -o ref-fft util.c fpr.c fft.c ref-fft.c -O0 -g3; 
# ./ref-fft | md5sum - 

set -e 
while inotifywait -e close_write fft.c; do 
    gcc -o ref-fft util.c fpr.c fft.c ref-fft.c -O0 -g3
    ./ref-fft
    ./ref-fft | md5sum -
done