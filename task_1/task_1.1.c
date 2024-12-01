#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void InputSize(int *rows, int *cols, int my_rank) {
    if (my_rank == 0) {
        scanf("%d", rows);
        scanf("%d", cols);
    }
    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void InputVector(int *vector, int cols, int my_rank) {
    if (my_rank == 0) 
        for (int i = 0; i < cols; ++i)
            vector[i] = rand() % 10;
    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);
}

void PrintVector(int *vector, int cols, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < cols; ++i) {
            printf("%d", vector[i]);
            printf("\n");
        }
    }
}

void InputMat(int *mat, int rows, int cols, int my_rank, int local_rows) {
    if (my_rank == 0) {
        int *temp = calloc(rows * cols, sizeof(int));
        for (int i = 0; i < cols * rows; ++i)
            temp[i] = rand() % 10;
        MPI_Scatter(temp, local_rows * cols, MPI_INT, mat, local_rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
        free(temp);
    } else {
    MPI_Scatter(NULL, local_rows * cols, MPI_INT, mat, local_rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void PrintMat(int rows, int cols, int *local_mat, int my_rank, int local_rows) {
    if (my_rank == 0) {
        int *temp = calloc(rows * cols, sizeof(int));
        MPI_Gather(local_mat, local_rows * cols, MPI_INT, temp, local_rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                printf("%d ", temp[i * cols + j]);
            printf("\n");
        }
        free(temp);
    } else {
        MPI_Gather(local_mat, local_rows * cols, MPI_INT, NULL, local_rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void MatVecMul(int* mat, int* vec, int *res, int local_r, int cols) {
    for (int i = 0; i < local_r; ++i) {
        res[i] = 0;
        for (int j = 0; j < cols; ++j) { 
            res[i] += mat[i * cols + j] * vec[j];
        }
    }
}

void PrintRes(int *vector, int rows, int my_rank, int comm_size) {
    int *temp = calloc(rows, sizeof(int));
    if (my_rank == 0) {
        MPI_Gather(vector, rows / comm_size, MPI_INT, temp, rows / comm_size, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < rows; ++i) 
            printf("%d ", temp[i]);
        printf("\n");
    } else {
        MPI_Gather(vector, rows / comm_size, MPI_INT, NULL, rows / comm_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

int main() {
    int comm_size;
    int my_rank;
    
    double start, end, duration, max_duration;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    int rows, cols;

    InputSize(&rows, &cols, my_rank);


    int *mat = calloc(rows / comm_size * cols, sizeof(int)); 
    int *vec = calloc(cols, sizeof(int)); 
    InputVector(vec, cols, my_rank);
   // PrintVector(vec, cols, my_rank);
    InputMat(mat, rows, cols, my_rank, rows / comm_size);
    int *res = calloc(rows, sizeof(int)); 
  //  PrintMat(rows, cols, mat, my_rank, rows / comm_size);
    start =  MPI_Wtime();
    MatVecMul(mat, vec, res, rows / comm_size, cols);
    end =  MPI_Wtime();
    duration = end - start;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 //   PrintRes(res, rows, my_rank, comm_size);
    if (my_rank == 0)
        printf("Time: %f s\n", max_duration);
    MPI_Finalize();
}