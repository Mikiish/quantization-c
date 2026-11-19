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

void dequantize_array(
    const int8_t *input,
    float *output,
    size_t n,
    float scale
){
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] * scale;
    }
}

typedef struct {
    double max_error;
    double mse;
    double relative_error;
} ErrorMetrics;

ErrorMetrics compute_errors(
    const float *original,
    const float *reconstructed,
    size_t n
){
    ErrorMetrics metrics = {0.0f, 0.0, 0.0};
    if (n == 0) {
        return metrics;
    }
    double diff = (double)original[0] - reconstructed[0];
    double mse_error = diff * diff;
    double norme = (double)original[0] * original[0];
    double max_error = fabs(diff);
    for (size_t i = 1; i < n; ++i) {
        double diff = (double)original[i] - reconstructed[i];
        mse_error += diff * diff;
        norme += (double)original[i] * original[i];
        if (fabs(diff) > max_error) {
            max_error = fabs(diff);
        }
    }
    metrics.max_error = max_error;
    metrics.mse = mse_error / n;
    if (norme == 0.0) {
        metrics.relative_error = (mse_error == 0.0) ? 0.0 : INFINITY;
    } else {
        metrics.relative_error = sqrt(mse_error / norme);
    }
    return metrics;
}

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;



int main(void) {
    float input[5] = {-4.0f, -2.0f, 0.0f, 1.0f, 4.0f};
    int8_t output[5];
    size_t n = sizeof(input) / sizeof(input[0]);
    float scale = quant_scale(input, n);
    quantize_array(input, output, n, scale);
    float dequantized[5];
    dequantize_array(output, dequantized, n, scale);
    for (size_t i = 0; i < n; i++)
        printf("Original: %f, Quantized: %d, Dequantized: %f\n", input[i], output[i], dequantized[i]);
    ErrorMetrics metrics = compute_errors(input, dequantized, n);
    printf("Max absolute error: %f\n", metrics.max_error);
    printf("MSE: %f\n", metrics.mse);
    printf("Relative error: %f\n", metrics.relative_error);
    return 0;
}