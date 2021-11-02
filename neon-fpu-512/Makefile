# This Makefile compiles the implementation in this directory along with
# the known answer tests generator located in the
# ../../../KAT/generator/ directory. The output is an executable program
# in the build/ subdirectory, whose name starts with 'kat', followed by
# the implementation name (e.g. 'kat512int' for the 'falcon512int'
# implementation). This program, when executed, generates the .req and
# .rsp files in the expected NIST format.

.POSIX:

# CC = c99
CC = clang
CFLAGS = -W -Wall -O2 -Wextra -Wpedantic
LD = c99
LDFLAGS = 
LIBS = 

OBJ1 = build/codec.o build/util.o build/common.o build/fft.o build/fft_tree.o build/fpr.o build/keygen.o build/nist.o build/rng.o build/poly.o build/sign_sampler.o build/shake.o build/sign.o build/vrfy.o

OBJ2 = build/PQCgenKAT_sign.o build/katrng.o

HEAD1 = api.h fpr.h inner.h
HEAD2 = api.h katrng.h

all: build build/kat512fpu

build:
	-mkdir build

clean:
	-rm -f build/kat512fpu $(OBJ1) $(OBJ2)

build/kat512fpu: $(OBJ1) $(OBJ2)
	$(LD) $(LDFLAGS) -o build/kat512fpu $(OBJ1) $(OBJ2) $(LIBS)

build/codec.o: codec.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/codec.o codec.c

build/common.o: common.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/common.o common.c

build/fft.o: fft.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/fft.o fft.c

build/fpr.o: fpr.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/fpr.o fpr.c

build/keygen.o: keygen.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/keygen.o keygen.c

build/nist.o: nist.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/nist.o nist.c

build/rng.o: rng.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/rng.o rng.c

build/shake.o: shake.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/shake.o shake.c

build/sign.o: sign.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/sign.o sign.c

build/poly.o: poly.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/poly.o poly.c

build/util.o: util.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/util.o util.c

build/sign_sampler.o: sign_sampler.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/sign_sampler.o sign_sampler.c

build/fft_tree.o: fft_tree.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/fft_tree.o fft_tree.c

build/vrfy.o: vrfy.c $(HEAD1)
	$(CC) $(CFLAGS) -c -o build/vrfy.o vrfy.c

build/PQCgenKAT_sign.o: PQCgenKAT_sign.c $(HEAD2)
	$(CC) $(CFLAGS) -I . -DALGNAME=falcon512fpu -c -o build/PQCgenKAT_sign.o PQCgenKAT_sign.c

build/katrng.o: katrng.c $(HEAD2)
	$(CC) $(CFLAGS) -I . -c -o build/katrng.o katrng.c

