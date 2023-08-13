#include "api.h"
#include "randombytes.h"
#include <stdio.h>
#include <stdlib.h>

static void print(uint8_t *buf, size_t length, const char *string)
{
    printf("%s = [", string);
    for (size_t i = 0; i < length; i++)
    {
        printf("%02x", buf[i]);
    }
    printf("]\n");
}

static void fullcycle(uint8_t *public_key, uint8_t *secret_key,
                      uint8_t *signature, size_t signature_len,
                      uint8_t *message, size_t message_len, unsigned count)
{
    int ret = 0;
    ret = PQCLEAN_FALCON512_AARCH64_crypto_sign_keypair(public_key, secret_key);
    if (ret)
    {
        print(public_key, PQCLEAN_FALCON512_AARCH64_CRYPTO_PUBLICKEYBYTES, "public_key");
        print(secret_key, PQCLEAN_FALCON512_AARCH64_CRYPTO_SECRETKEYBYTES,
              "secret_key");
        printf("keygen error. Exiting. %u\n", count);
        exit(ret);
    }
    ret = PQCLEAN_FALCON512_AARCH64_crypto_sign_signature(
            signature, &signature_len, message, message_len, secret_key);
    if (ret)
    {
        print(public_key, PQCLEAN_FALCON512_AARCH64_CRYPTO_PUBLICKEYBYTES, "public_key");
        print(secret_key, PQCLEAN_FALCON512_AARCH64_CRYPTO_SECRETKEYBYTES,
              "secret_key");
        print(signature, signature_len, "signature");
        printf("sign error. Exiting. %u\n", count);
        exit(ret);
    }
    ret = PQCLEAN_FALCON512_AARCH64_crypto_sign_verify(
            signature, signature_len, message, message_len, public_key);
    if (ret)
    {
        printf("verify error. Exiting. %u\n", count);
        exit(ret);
    }
}

int main(void)
{

    uint8_t *public_key = NULL;
    uint8_t *secret_key = NULL;
    uint8_t *signature = NULL;
    uint8_t message[50];
    size_t message_len = 50;
    size_t signature_len = 0;

    public_key = malloc(PQCLEAN_FALCON512_AARCH64_CRYPTO_PUBLICKEYBYTES);
    secret_key = malloc(PQCLEAN_FALCON512_AARCH64_CRYPTO_SECRETKEYBYTES);
    signature = malloc(PQCLEAN_FALCON512_AARCH64_CRYPTO_BYTES);


    srand(1234);
    randombytes(message, message_len);
    print(message, message_len, "msg");

    unsigned count = 0;
    while (1)
    {
        fullcycle(public_key, secret_key, signature, signature_len, message,
                  message_len, count);
        count++;
    }

    free(public_key);
    free(secret_key);
    free(signature);

    return 0;
}
