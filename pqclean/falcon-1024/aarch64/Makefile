# This Makefile can be used with GNU Make or BSD Make

LIB=libfalcon-512_clean.a

SOURCES = codec.c keygen.c poly_float.c common.c fft_tree.c ntt_consts.c poly_int.c sign.c fpr.c ntt.c rng.c util.c fft.c sampler.c vrfy.c ../../../common/randombytes.c ../../../common/fips202.c pqclean.c
OBJECTS = codec.o keygen.o poly_float.o common.o fft_tree.o ntt_consts.o poly_int.o sign.o fpr.o ntt.o rng.o util.o fft.o sampler.o vrfy.o ../../../common/randombytes.o ../../../common/fips202.o pqclean.o
HEADERS = api.h fpr.h inner.h macrof.h macrofx4.h macrous.h ntt_consts.h params.h poly.h util.h

CFLAGS=-std=c99 -O3 -W -Wall -Wconversion -Wextra -Wpedantic -Wvla -Werror -Wmissing-prototypes -Wredundant-decls -I../../../common $(EXTRAFLAGS)

all: $(LIB)
test: test_api.o test_api

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

test_api.o: test_api.c
	$(CC) $(CFLAGS) -c -o $@ $<

test_api: $(HEAD) $(OBJECTS) test_api.c
	$(CC) $(CFLAGS) -o $@ $(OBJECTS) test_api.o
	./$@

$(LIB): $(OBJECTS)
	$(AR) -r $@ $(OBJECTS)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(LIB)
