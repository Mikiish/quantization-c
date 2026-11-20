#include <stddef.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <stdexcept>
#include <chrono>
using namespace std;

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

template <typename T>
struct Matrix {
    size_t rows;
    size_t cols;
    T *data;

    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(nullptr) {
        data = new T[rows * cols];
    }
    ~Matrix() {
        delete[] data;
    }
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&& other) noexcept
        : rows(other.rows), cols(other.cols), data(other.data) {
        other.rows = 0;
        other.cols = 0;
        other.data = nullptr;
    }
    T& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }
    const T& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            printf("[ ");
            for (size_t j = 0; j < cols; ++j) {
                printf("%f ", static_cast<double>(data[i * cols + j]));
            }
            printf("]\n");
        }
    }
};

Matrix<float> create_vandermonde_matrix(vector<float> &vec) {
    size_t n = vec.size();
    Matrix<float> mat(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mat(i, j) = powf(vec[i], (float)j);
        }
    }
    return mat;
}

struct Q8Matrix {
    Matrix<int8_t> mat;
    float scale;
    Q8Matrix(size_t rows, size_t cols, float scale)
        : mat(rows, cols), scale(scale) {}
};

Q8Matrix quantize_matrix(const Matrix<float> &input) {
    size_t rows = input.rows;
    size_t cols = input.cols;
    float scale = quant_scale(input.data, rows * cols);
    Q8Matrix qmat(rows, cols, scale);
    quantize_array(input.data, qmat.mat.data, rows * cols, scale);
    return qmat;
}

Matrix<float> dequantize_matrix(const Q8Matrix &qmat) {
    size_t rows = qmat.mat.rows;
    size_t cols = qmat.mat.cols;
    Matrix<float> mat(rows, cols);
    dequantize_array(qmat.mat.data, mat.data, rows * cols, qmat.scale);
    return mat;
}

template<typename T, typename Acc>
Matrix<Acc> matmul(
    const Matrix<T>& A,
    const Matrix<T>& B
) {
    if (A.cols != B.rows) {
        throw invalid_argument("⛔ Matrix dimensions do not match for multiplication.");
    }
    Matrix<Acc> C(A.rows, B.cols);
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            Acc sum = 0;
            for (size_t k = 0; k < A.cols; ++k) {
                sum += static_cast<Acc>(A(i, k)) * static_cast<Acc>(B(k, j));
            }
            C(i, j) = sum;
        }
    }
    return C;
}

/* Remarque : INT8 vaut 127 au max et 2^31-1 pour un INT32, le coef maximal d'un produit de matrice quantifiée est n*127^2.
    On peut donc multiplier des matrices de taille n < 2^31/(127^2) = 133169. On est donc très loin d'avoir un overflow sur un int32_t.
*/
Matrix<float> quantized_matmul(
    const Q8Matrix& QA,
    const Q8Matrix& QB
) {
    Matrix<int32_t> S = matmul<int8_t, int32_t>(QA.mat, QB.mat); // Cette matrice n'est PAS la quantifiée du produit !
    Matrix<float> C(S.rows, S.cols);
    float scale = QA.scale * QB.scale;
    for (size_t i = 0; i < S.rows; ++i) {
        for (size_t j = 0; j < S.cols; ++j) {
            C(i, j) = S(i, j) * scale;
        }
    }
    return C;
}


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

    vector<float> vec = {-4.0f, -2.0f, 0.0f, 1.0f, 4.0f};
    Matrix<float> mat_vandermonde = create_vandermonde_matrix(vec);
    mat_vandermonde.print();
    Q8Matrix qmat = quantize_matrix(mat_vandermonde);
    printf("Quantized Vandermonde Matrix:\n");
    qmat.mat.print();
    Matrix<float> dequantized_mat = dequantize_matrix(qmat);
    printf("Dequantized Vandermonde Matrix:\n");
    dequantized_mat.print();
    ErrorMetrics mat_metrics = compute_errors(mat_vandermonde.data, dequantized_mat.data, mat_vandermonde.rows * mat_vandermonde.cols);
    printf("Matrix Max absolute error: %f\n", mat_metrics.max_error);
    printf("Matrix MSE: %f\n", mat_metrics.mse);
    printf("Matrix Relative error: %f\n", mat_metrics.relative_error);

    Matrix<float> A(2, 2);
    A(0, 0) = 1.0f; A(0, 1) = 2.0f; A(1, 0) = 3.0f; A(1, 1) = 4.0f;
    Matrix<float> B(2, 2);
    B(0, 0) = 5.0f; B(0, 1) = 6.0f; B(1, 0) = 7.0f; B(1, 1) = 8.0f;
    Matrix<float> C = matmul<float, float>(A, B);
    printf("Matrix Product:\n");
    C.print();
    Q8Matrix QA = quantize_matrix(A);
    Q8Matrix QB = quantize_matrix(B);
    Matrix<float> C_quant = quantized_matmul(QA, QB);
    printf("Quantized Matrix Product:\n");
    C_quant.print();
    ErrorMetrics comp = compute_errors(C.data, C_quant.data, C.rows * C.cols);
    printf("Quantized Matrix Product Max absolute error: %f\n", comp.max_error);
    printf("Quantized Matrix Product MSE: %f\n", comp.mse);
    printf("Quantized Matrix Product Relative error: %f\n", comp.relative_error);

    return 0;
}