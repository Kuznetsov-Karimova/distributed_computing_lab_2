#ifndef _MATRIX_UTIL_H_
#define _MATRIX_UTIL_H_

#include <stdlib.h>
#include <stdio.h>

void init_matrix(int ***matrix, int matrixSizeN) {
    // создаём один большой массив для хранения всех элементов матрицы в памяти
    // вместо того, чтобы выделять память для каждого ряда отдельно
    int *p = (int *)malloc(sizeof(int) * matrixSizeN * matrixSizeN);
    if (p == NULL) {
        printf("Error allocating memory\n");
        return;
    }
    *matrix = (int **)malloc(matrixSizeN * sizeof(int *));
    if (matrix == NULL) {
        free(p);
        printf("Error allocating memory\n");
        return;
    }
    for (int i = 0; i < matrixSizeN; ++i) {
        (*matrix)[i] = &(p[i * matrixSizeN]);
    }
}

void free_matrix(int ***matrix) {
    free(&((*matrix)[0][0]));
    free(*matrix);
}

void multiply_matrices(int **matA, int **matB, int matrixSizeN, int ***resMat) {
    for (int i = 0; i < matrixSizeN; ++i) {
        for (int j = 0; j < matrixSizeN; ++j) {
            int result = 0;
            for (int k = 0; k < matrixSizeN; ++k) {
                result += matA[i][k] * matB[k][j];
            }
            (*resMat)[i][j] = result;
        }
    }
}

void print_matrix_to_screen(int **matrix, int matrixSizeN) {
    for (int i = 0; i < matrixSizeN; ++i) {
        for (int j = 0; j < matrixSizeN; ++j) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_matrix_to_file(int **matrix, int matrixSizeN, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening a file for recording\n");
        return;
    }

    for (int i = 0; i < matrixSizeN; ++i) {
        for (int j = 0; j < matrixSizeN; ++j) {
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void fill_matrix_randomly(int **matrix, int matrixSizeN, unsigned int seed) {
    srand(seed); 

    for (int i = 0; i < matrixSizeN; ++i) {
        for (int j = 0; j < matrixSizeN; ++j) {
            matrix[i][j] = rand() % 10;
        }
    }
}

#endif