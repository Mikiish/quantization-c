#include <stddef.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

float max_abs(const float *data, size_t n) {
    if (n == 0) {
        return 0.0f; // En cas de tableau vide, retourner 0.0f.
    }
    float max = fabsf(data[0]);
    for (size_t i = 1; i < n; ++i) {
        float abs_value = fabsf(data[i]);
        if (abs_value > max) {
            max = abs_value;
        }
    }
    return max;
}

float quant_scale(const float *data, size_t n) {
    float max_value = max_abs(data, n);
    if (max_value == 0.0f) {
        return 1.0f; // Éviter la division par zéro, retourner 1.0f.
    }
    return max_value / 127.0f; // Échelle pour quantification sur 8 bits.
}

int8_t quantize_scalar(float x, float scale) {
    float q = roundf(x / scale);
    q = fmaxf(q, -127.0f); // Limiter à l'intervalle [-127, 127]
    q = fminf(q, 127.0f); // Limiter à l'intervalle [-127, 127]
    return (int8_t)q;
}

void quantize_array(
    const float *input,
    int8_t *output,
    size_t n,
    float scale
){
    for (size_t i = 0; i < n; i++) {
        output[i] = quantize_scalar(input[i], scale);
    }
}

int main(void) {
    float input[5] = {-4.0f, -2.0f, 0.0f, 1.0f, 4.0f};
    int8_t output[5];
    size_t n = sizeof(input) / sizeof(input[0]);
    float scale = quant_scale(input, n);
    quantize_array(input, output, n, scale);
    for (size_t i = 0; i < n; i++)
        printf("Scale: %d\n", (int)output[i]);

    return 0;
}