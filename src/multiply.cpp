#include "multiply.h"

// TODO: you should implement your code in this file, we will only call `matrix_multiplication` to 
// test your implementation

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P])
{
    for(int row = 0; row < N; ++row)
            for(int col = 0; col < P; ++col)
                for(int mid = 0; mid < M; ++mid)
                    result_matrix[row][col] += matrix1[row][mid] * matrix2[mid][col];
}
