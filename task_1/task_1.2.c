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

void InputVector(int *vector, int cols, int my_rank, int local_c) {
    int *temp = calloc(cols, sizeof(int));
    if (my_rank == 0) {
        for (int i = 0; i < cols; ++i)
            temp[i] = rand() % 10;
        MPI_Scatter(temp, local_c, MPI_INT, vector, local_c, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
    MPI_Scatter(NULL, local_c, MPI_INT, vector, local_c, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void PrintVector(int *vector, int cols, int my_rank,  int local_cols) {
    int *temp = calloc(cols, sizeof(int));
    if (my_rank == 0) {
        MPI_Gather(vector, local_cols, MPI_INT, temp, local_cols, MPI_INT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < cols; ++i) 
            printf("%d ", temp[i]);
        printf("\n");
    } else {
        MPI_Gather(vector, local_cols, MPI_INT, NULL, local_cols, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void InputMat(int *mat, int rows, int cols, int my_rank, int local_cols, MPI_Datatype *col_type) {
    if (my_rank == 0) {
        int *temp = calloc(rows * cols, sizeof(int));
        for (int i = 0; i < cols * rows; ++i)
            temp[i] = rand() % 10;
        MPI_Scatter(temp, 1, *col_type, mat, local_cols * rows, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(NULL, 1, *col_type, mat, local_cols * rows, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void PrintMat(int rows, int cols, int *local_mat, int my_rank, int local_cols,  MPI_Datatype *col_type) {
    if (my_rank == 0) {
        int *temp = calloc(rows * cols, sizeof(int));
        MPI_Gather(local_mat, local_cols * rows, MPI_INT, temp, 1, *col_type, 0, MPI_COMM_WORLD);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                printf("%d ", temp[i * cols + j]);
            printf("\n");
        }
        free(temp);
    } else {
        MPI_Gather(local_mat, local_cols * rows, MPI_INT, NULL, 1, *col_type, 0, MPI_COMM_WORLD);
    }
}

void MatVecMul(int* mat, int* vec, int *res, int local_cols, int rows, int cols, int my_rank) {
    for (int i = 0; i < rows; ++i) {
        res[i] = 0;
        for (int j = 0; j < local_cols; ++j) { 
            res[i] += mat[i * local_cols + j] * vec[j];
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

float main() {
    int comm_size;
    int my_rank;
    double start, end, duration, max_duration;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int rows, cols;

    InputSize(&rows, &cols, my_rank);
    int local_cols = cols / comm_size;

    MPI_Datatype column, col_type;
    MPI_Type_vector(rows, local_cols, cols, MPI_INT, &column);
    MPI_Type_commit(&column);
    MPI_Type_create_resized(column, 0, local_cols * sizeof(int), &col_type);
    MPI_Type_commit(&col_type);

    int *mat = calloc(cols / comm_size * rows, sizeof(int)); 
    int *vec = calloc(cols / comm_size, sizeof(int)); 
    InputVector(vec, cols, my_rank, cols / comm_size);
    // PrintVector(vec, cols, my_rank, cols / comm_size);
    InputMat(mat, rows, cols, my_rank, cols / comm_size, &col_type);
    int *res = calloc(rows, sizeof(int)); 
   // PrintMat(rows, cols, mat, my_rank, cols / comm_size, &col_type);
    start =  MPI_Wtime();
    MatVecMul(mat, vec, res, cols / comm_size, rows, cols, my_rank);
    int *res_final = calloc(rows, sizeof(int)); 
    MPI_Reduce(res, res_final, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
   // PrintRes(res_final, rows, my_rank, comm_size);

    end =  MPI_Wtime();
    duration = end - start;
    MPI_Reduce(&duration, &max_duration,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("Time: %f s\n", max_duration);
    MPI_Finalize();
}