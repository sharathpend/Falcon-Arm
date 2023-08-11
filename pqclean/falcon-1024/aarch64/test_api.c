#include "api.h"
#include "randombytes.h"
#include <stdio.h>
#include <stdlib.h>

static
void print(uint8_t *buf, size_t length, const char *string) {
  printf("%s = [", string);
  for (size_t i = 0; i < length; i++) {
    printf("%02x", buf[i]);
  }
  printf("]\n");
}

static void fullcycle(uint8_t *public_key, uint8_t *secret_key,
                      uint8_t *signature, size_t signature_len,
                      uint8_t *message, size_t message_len, unsigned count) {

  if (PQCLEAN_FALCON1024_CLEAN_crypto_sign_keypair(public_key, secret_key)) {
    printf("keygen error. Exiting. %u\n", count);
    exit(-1);
  }
  if (PQCLEAN_FALCON1024_CLEAN_crypto_sign_signature(
          signature, &signature_len, message, message_len, secret_key)) {
    print(signature, signature_len, "signature");
    print(secret_key, PQCLEAN_FALCON1024_CLEAN_CRYPTO_SECRETKEYBYTES, "secret_key");
    printf("sign error. Exiting. %u\n", count);
    exit(-1);
  }
  if (PQCLEAN_FALCON1024_CLEAN_crypto_sign_verify(
          signature, signature_len, message, message_len, public_key)) {
    printf("verify error. Exiting. %u\n", count);
    exit(-1);
  }
//   if (count == 170) {
//     print(public_key, PQCLEAN_FALCON1024_CLEAN_CRYPTO_PUBLICKEYBYTES, "public_key");
//     print(secret_key, PQCLEAN_FALCON1024_CLEAN_CRYPTO_SECRETKEYBYTES, "secret_key");
//     // print(signature, signature_len, "signature");
//     exit(0);
//   }
}

int main(void) {

  uint8_t *public_key = NULL;
  uint8_t *secret_key = NULL;
  uint8_t *signature = NULL;
  uint8_t message[50];
  uint8_t entropy_input[48];
  size_t message_len = 50;
  size_t signature_len = 0;

  public_key = malloc(PQCLEAN_FALCON1024_CLEAN_CRYPTO_PUBLICKEYBYTES);
  secret_key = malloc(PQCLEAN_FALCON1024_CLEAN_CRYPTO_SECRETKEYBYTES);
  signature = malloc(PQCLEAN_FALCON1024_CLEAN_CRYPTO_BYTES);

  for (uint8_t i = 0; i < 48; i++) {
    entropy_input[i] = i;
  }

//   randombytes_init(entropy_input, NULL, 256);
  randombytes(message, message_len);

  unsigned count = 0;
  while (1) {
    printf(".");
    fflush(stdout);
    fullcycle(public_key, secret_key, signature, signature_len, message,
              message_len, count);
    count++;
  }

  free(public_key);
  free(secret_key);
  free(signature);

  return 0;
}
