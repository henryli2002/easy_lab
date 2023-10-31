#include "multiply.h"
#include "string.h"
#include <thread>
#include <vector>
#include <emmintrin.h> // SSE intrinsics


// TODO: you should implement your code in this file, we will only call `matrix_multiplication` to 
// test your implementation

// // 简单三层循环
// void matrix_multiplication(double matrix1[N][M], 
//                             double matrix2[M][P], 
//                             double result_matrix[N][P])
// {
//     for(int row = 0; row < N; ++row)
//         for(int col = 0; col < P; ++col) {
//             double a = 0;
//             for(int mid = 0; mid < M; ++mid) {
//                 a += matrix1[row][mid] * matrix2[mid][col];
//             }
//             result_matrix[row][col] = a;
//         }
// }

// //交换计算顺序
// void matrix_multiplication(double matrix1[N][M], 
//                             double matrix2[M][P], 
//                             double result_matrix[N][P])
// {
//     for(int row = 0; row < N; ++row)
//         for(int mid = 0; mid < M; ++mid) {
//             for(int col = 0; col < P; ++col) {
//                 result_matrix[row][col] += matrix1[row][mid] * matrix2[mid][col];
//             } 
//         }
// }

// // 矩阵转置
// double transposed[P][M];
// void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < P; ++j) {
//             transposed[j][i] = matrix2[i][j];
//         }
//     }
//     for (int row = 0; row < N; row++) {
//         for (int col = 0; col < P; col++) {
//             double a = 0;
//             for (int mid = 0; mid < M; mid++) {
//                 a += matrix1[row][mid] * transposed[col][mid];
//             }
//             result_matrix[row][col] = a;
//         }
//     }
// }

// 多线程
const int num_threads = 30;  // 线程数量
double transposed[num_threads][P][M];

// 函数用于计算部分矩阵乘法
void multiply_partial(int start_row, 
                        int end_row, 
                        double matrix1[N][M], 
                        double matrix2[M][P], 
                        double result_matrix[N][P], 
                        double transposed_thread[P][M]) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            transposed_thread[j][i] = matrix2[i][j];
        }
    }
    for (int row = start_row; row < end_row; ++row) {
        for (int col = 0; col < P; ++col) {
            double a = 0;
            for (int mid = 0; mid < M; ++mid) {
                a += matrix1[row][mid] * transposed_thread[col][mid];
            }
            result_matrix[row][col] = a;
        }
    }
}

void multiply_partial_sse(int start_row, 
                          int end_row, 
                          double matrix1[N][M], 
                          double matrix2[M][P], 
                          double result_matrix[N][P], 
                          double transposed_thread[P][M]) {
    // Transpose the second matrix
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            transposed_thread[j][i] = matrix2[i][j];
        }
    }

    for (int row = start_row; row < end_row; ++row) {
        for (int col = 0; col < P; ++col) {
            __m128d sum = _mm_setzero_pd();  // Initialize sum to zero

            // We'll unroll the loop by a factor of 2 to process two doubles at a time
            for (int mid = 0; mid < M; mid += 2) {
                __m128d vec1 = _mm_loadu_pd(&matrix1[row][mid]);
                __m128d vec2 = _mm_loadu_pd(&transposed_thread[col][mid]);
                sum = _mm_add_pd(sum, _mm_mul_pd(vec1, vec2));
            }

            // Handle case where M is not divisible by 2
            double partial_sum[2];
            _mm_storeu_pd(partial_sum, sum);
            double final_sum = partial_sum[0] + partial_sum[1];
            if (M % 2 != 0) {
                final_sum += matrix1[row][M - 1] * transposed_thread[col][M - 1];
            }

            result_matrix[row][col] = final_sum;
        }
    }
}

// 函数用于并行计算矩阵乘法
void matrix_multiplication(double matrix1[N][M], 
                            double matrix2[M][P], 
                            double result_matrix[N][P]) {
    std::vector<std::thread> threads;

    // 每个线程计算的行数
    int rows_per_thread = N / num_threads;

    // 启动线程并计算
    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? N : (i + 1) * rows_per_thread;
        threads.push_back(std::thread(multiply_partial_sse,  \
        start_row, end_row, matrix1, matrix2, result_matrix, transposed[i]));
    }

    // 等待所有线程完成
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
}
