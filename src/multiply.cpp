#include "multiply.h"
#include <iostream>
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

// // 多线程
// const int num_threads = 24;  // 线程数量
// double transposed[P][M];

// // 函数用于计算部分矩阵乘法
// void multiply_partial(int start_row, 
//                         int end_row, 
//                         double matrix1[N][M],
//                         double result_matrix[N][P]) {
//     for (int row = start_row; row < end_row; ++row) {
//         for (int col = 0; col < P; ++col) {
//             double a = 0;
//             for (int mid = 0; mid < M; ++mid) {
//                 a += matrix1[row][mid] * transposed[col][mid];
//             }
//             result_matrix[row][col] = a;
//         }
//     }
// }

// void multiply_partial_sse(int start_row, 
//                           int end_row, 
//                           double matrix1[N][M],
//                           double result_matrix[N][P]) {

//     for (int row = start_row; row < end_row; ++row) {
//         for (int col = 0; col < P; ++col) {
//             __m128d sum = _mm_setzero_pd();  // Initialize sum to zero

//             // We'll unroll the loop by a factor of 2 to process two doubles at a time
//             for (int mid = 0; mid < M; mid += 2) {
//                 __m128d vec1 = _mm_loadu_pd(&matrix1[row][mid]);
//                 __m128d vec2 = _mm_loadu_pd(&transposed[col][mid]);
//                 sum = _mm_add_pd(sum, _mm_mul_pd(vec1, vec2));
//             }

//             // Handle case where M is not divisible by 2
//             double partial_sum[2];
//             _mm_storeu_pd(partial_sum, sum);
//             double final_sum = partial_sum[0] + partial_sum[1];
//             if (M % 2 != 0) {
//                 final_sum += matrix1[row][M - 1] * transposed[col][M - 1];
//             }

//             result_matrix[row][col] = final_sum;
//         }
//     }
// }

// // 函数用于并行计算矩阵乘法
// void matrix_multiplication(double matrix1[N][M], 
//                             double matrix2[M][P], 
//                             double result_matrix[N][P]) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < P; ++j) {
//             transposed[j][i] = matrix2[i][j];
//         }
//     }
//     std::vector<std::thread> threads;
//     if (N < num_threads) {
//         multiply_partial(0, N, matrix1, result_matrix);
//         return;
//     }
//     // 每个线程计算的行数
//     int rows_per_thread = N / num_threads;
    
//     // 启动线程并计算
//     for (int i = 0; i < num_threads; ++i) {
//         int start_row = i * rows_per_thread;
//         int end_row = (i == num_threads - 1) ? N : (i + 1) * rows_per_thread;
//         threads.push_back(std::thread(multiply_partial,  \
//         start_row, end_row, matrix1, result_matrix));
//     }    
    

//     // 等待所有线程完成
//     for (int i = 0; i < num_threads; ++i) {
//         threads[i].join();
//     }
// }

#include <emmintrin.h>  // for SSE2
#include <vector>
#include <thread>

const int num_threads = 40;  // 线程数量

double* transposed;
double* matrix1_1;
double* result_matrix_1;
// 使用SSE进行计算的函数
void multiply_partial_sse(int start_row, int end_row) {
    for (int row = start_row; row < end_row; ++row) {
        for (int col = 0; col < P; ++col) {
            __m128d sum = _mm_setzero_pd();

            for (int mid = 0; mid < M; mid += 2) {
                __m128d vec1 = _mm_load_pd(&matrix1_1[row * M + mid]);
                __m128d vec2 = _mm_load_pd(&transposed[col * M + mid]);
                sum = _mm_add_pd(sum, _mm_mul_pd(vec1, vec2));
            }

            double partial_sum[2];
            _mm_storeu_pd(partial_sum, sum);
            double final_sum = partial_sum[0] + partial_sum[1];
            if (M % 2 != 0) {
                final_sum += matrix1_1[row * M + M - 1] * transposed[col * M + M - 1];
            }

            result_matrix_1[row * P + col] = final_sum;
        }
    }
}

void multiply_partial(int start_row, int end_row) {
    for (int row = start_row; row < end_row; ++row) {
        for (int col = 0; col < P; ++col) {
            double a = 0;
            for (int mid = 0; mid < M; ++mid) {
                a += matrix1_1[row * M + mid] * transposed[col * M + mid];
            }
            result_matrix_1[row * P + col] = a;
        }
    }
}

void matrix_multiplication(double matrix1[N][M], 
                            double matrix2[M][P], 
                            double result_matrix[N][P]) {
    transposed = (double*)_mm_malloc(sizeof(double) * M * P, 16);
    matrix1_1 = (double*)_mm_malloc(N * M * sizeof(double), 16);
    result_matrix_1 = (double*)_mm_malloc(N * P * sizeof(double), 16);

    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            matrix1_1[i * M + j] = matrix1[i][j];
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            transposed[j * M + i] = matrix2[i][j];
        }
    }
    
    std::vector<std::thread> threads;
    int rows_per_thread = N / num_threads;
    
    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? N : (i + 1) * rows_per_thread;
        if(i % 2 == 0){
            threads.push_back(std::thread(multiply_partial_sse, start_row, end_row));
        }
        else{
            threads.push_back(std::thread(multiply_partial, start_row, end_row));
        }
    }    

    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            result_matrix[i][j] = result_matrix_1[i * P + j];
        }
    }
    
    _mm_free(transposed);
    _mm_free(matrix1_1);
    _mm_free(result_matrix_1);
}
