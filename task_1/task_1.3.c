#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

void InputSize(int *rows, int *cols, int my_rank) {
    if (my_rank == 0) {
        scanf("%d", rows);
        scanf("%d", cols);
    }
    MPI_Bcast(rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void InputVector(int *vector, int cols, int my_rank) {
    if (my_rank == 0) {
        for (int i = 0; i < cols; ++i)
            vector[i] = rand() % 10;
    }
    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);
}

void PrintVector(int *vector, int cols, int my_rank) {
    if (my_rank == 1) {
        for (int i = 0; i < cols; ++i) 
            printf("%d ", vector[i]);
        printf("\n");
    } 
}

void InputMat(int *mat, int rows, int cols, int my_rank, int local_cols, MPI_Datatype *col_type, int *disps, int *counts) {
    if (my_rank == 0) {
        int *temp = calloc(rows * cols, sizeof(int));
        for (int i = 0; i < cols * rows; ++i)
            temp[i] = rand() % 10;
        MPI_Scatterv(temp, counts, disps, *col_type, mat, local_cols * local_cols, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(NULL, counts, disps, *col_type, mat, local_cols * local_cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void PrintMat(int rows, int cols, int *local_mat, int my_rank, int local_cols,  MPI_Datatype *col_type, int *disps, int *counts) {
    if (my_rank == 0) {
        int *temp = calloc(rows * cols, sizeof(int));
        MPI_Gatherv(local_mat, local_cols * local_cols, MPI_INT, temp, counts, disps, *col_type, 0, MPI_COMM_WORLD);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                printf("%d ", temp[i * cols + j]);
            printf("\n");
        }
        free(temp);
    } else {
        MPI_Gatherv(local_mat, local_cols * local_cols, MPI_INT, NULL, counts, disps, *col_type, 0, MPI_COMM_WORLD);
    }
}

void MatVecMul(int* mat, int* vec, int *res, int local_cols, int rows, int cols, int my_rank, int block_in_row) {
    for (int i = 0; i < local_cols; ++i) {
        for (int j = 0; j < local_cols; ++j) { 
            res[i + local_cols * (my_rank / block_in_row)] += mat[i * local_cols + j] * vec[j + local_cols * (my_rank % block_in_row)];
        }
    }
}

void PrintRes(int *vector, int rows, int my_rank, int comm_size) {
    if (my_rank == 0) {
        for (int i = 0; i < rows; ++i) 
            printf("%d ", vector[i]);
        printf("\n");
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

    int block_in_row = sqrt(comm_size);
   // printf("block_in_row %d \n", block_in_row);
    int local_cols = cols / block_in_row;
    int local_rows = rows / block_in_row;

    MPI_Datatype block, block_type;

    MPI_Type_vector(local_rows, local_cols, cols, MPI_INT, &block);
    MPI_Type_create_resized(block, 0, sizeof(int), &block_type);
    MPI_Type_commit(&block_type);
    int disps[comm_size];
    int vec_disps[comm_size];
    int counts[comm_size];
    for (int i = 0; i < block_in_row; ++i) {
        for (int j = 0; j < block_in_row; ++j) {
            counts[i * block_in_row + j] = 1;
            disps[i * block_in_row + j] = i * cols * local_rows + j * local_cols;
        }
    }
    int *mat = calloc(local_cols * local_rows, sizeof(int)); 
    int *vec = calloc(cols , sizeof(int)); 
    InputVector(vec, cols, my_rank);
   // PrintVector(vec, cols, my_rank);
    InputMat(mat, rows, cols, my_rank, local_cols, &block_type, disps, counts);
    int *res = calloc(rows, sizeof(int)); 
    for (int i = 0; i < rows; ++i)
        res[i] = 0;
   // PrintMat(rows, cols, mat, my_rank, local_cols, &block_type, disps, counts);
    start =  MPI_Wtime();
    MatVecMul(mat, vec, res, local_cols, rows, cols, my_rank, block_in_row);
    int *res_final = calloc(rows, sizeof(int)); 
    MPI_Reduce(res, res_final, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  //  PrintRes(res_final, rows, my_rank, comm_size);

    end =  MPI_Wtime();
    duration = end - start;
    MPI_Reduce(&duration, &max_duration,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("Time: %f s\n", max_duration);

    MPI_Finalize();
}