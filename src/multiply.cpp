#include "multiply.h"
#include "string.h"
#include <thread>
#include <vector>

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
const int num_threads = 10;  // 线程数量

// 函数用于计算部分矩阵乘法
void multiply_partial(int start_row, int end_row, double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
    for (int row = start_row; row < end_row; ++row) {
        for (int col = 0; col < P; ++col) {
            double a = 0;
            for (int mid = 0; mid < M; ++mid) {
                a += matrix1[row][mid] * matrix2[mid][col];
            }
            result_matrix[row][col] = a;
        }
    }
}

// 函数用于并行计算矩阵乘法
void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {
    std::vector<std::thread> threads;

    // 每个线程计算的行数
    int rows_per_thread = N / num_threads;

    // 启动线程并计算
    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? N : (i + 1) * rows_per_thread;
        threads.push_back(std::thread(multiply_partial, start_row, end_row, matrix1, matrix2, result_matrix));
    }

    // 等待所有线程完成
    for (int i = 0; i < num_threads; ++i) {
        threads[i].join();
    }
}
