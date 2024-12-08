#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// библиотека с функциями для работы с матрицами
#include "matrix_util.h"

// матрицы для перемножения заполняются рандомно
void fill_matrices_randomly(int **matA, int **matB, int size) {
    unsigned int seedA = (unsigned int)(time(NULL) * getpid() + 1);
    unsigned int seedB = (unsigned int)(time(NULL) * getpid() + 2);
    fill_matrix_randomly(matA, size, seedA);
    fill_matrix_randomly(matB, size, seedB);
}

// глобальные переменные
int countOfP; // количество исполняющих процессов
int matrixSizeN; // размер матрицы(NxN)
int procGridDim; // количество процессов по каждому из измерений (по строкам и столбцам) в двумерной решетке
int blockSize; // размер блока каждой локальной матрицы, которая будет обрабатываться каждым процессом
int broadcastData[4]; // данные, которые передаются всем процессам для синхронизации

void checkParams_getParamsForBlocks() {
    double sqroot = sqrt(countOfP);
    if ((sqroot - floor(sqroot)) != 0) {
        // Число процессов быть полным квадратом
        printf("The number of processes should be a square\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // Исходные матрицы должны иметь размерность, кратную корню от количества процессов
    int intRoot = (int)sqroot;
    if (matrixSizeN % intRoot != 0) {
        printf("The initial matrices must have a dimension that is a multiple of the root of the number of processes\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    procGridDim = intRoot;
    blockSize = matrixSizeN / intRoot;
}

void cart_create(MPI_Comm* cartesianComm) {
    MPI_Bcast(&broadcastData, 4, MPI_INT, 0, MPI_COMM_WORLD);
    procGridDim = broadcastData[0];
    blockSize = broadcastData[1];
    matrixSizeN = broadcastData[2];
    matrixSizeN = broadcastData[3];

    int gridDim[2] = {procGridDim, procGridDim};

    int periods[2] = {1,1};

    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, gridDim, periods, reorder, cartesianComm);
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    matrixSizeN = atoi(argv[1]); // получаем размер матриц

    int rank; // идентификатор процесса

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &countOfP);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int **matA = NULL, **matB = NULL;
    int **resMat = NULL;

    if (rank == 0) {
        checkParams_getParamsForBlocks();

        init_matrix(&matA, matrixSizeN);

        init_matrix(&matB, matrixSizeN);

        unsigned int seedA = (unsigned int)(time(NULL) * getpid() + 1);
        fill_matrix_randomly(matA, matrixSizeN, seedA);

        // printf("A:\n");
        // print_matrix_to_screen(matA, matrixSizeN);
        // print_matrix_to_file(matA, matrixSizeN, "matA.txt");

        unsigned int seedB = (unsigned int)(time(NULL) * getpid() + 2);
        fill_matrix_randomly(matB, matrixSizeN, seedB);

        // printf("B:\n");
        // print_matrix_to_screen(matB, matrixSizeN);
        // print_matrix_to_file(matB, matrixSizeN, "matB.txt");

        init_matrix(&resMat, matrixSizeN);

        broadcastData[0] = procGridDim;
        broadcastData[1] = blockSize;
        broadcastData[2] = matrixSizeN;
        broadcastData[3] = matrixSizeN;
    }

    MPI_Comm cartesianComm; // коммуникатор для виртуальной решетки процессов, который используется для обмена данными между процессами
    // Процессы организуются в виртуальную декартову топологию
    cart_create(&cartesianComm);

    // Матрицы разбиваются на равное количество квадратных блоков
    int **localA = NULL, **localB = NULL;
    init_matrix(&localA, blockSize);
    init_matrix(&localB, blockSize);

    // Определение размеров глобальной и локальной матрицы
    int globalSize[2] = {matrixSizeN, matrixSizeN}; // Размеры глобальной матрицы
    int localSize[2] = {blockSize, blockSize}; // Размеры локальной матрицы
    int starts[2] = {0, 0}; // Начальные индексы для подмассива
    
    MPI_Datatype type, subarrtype;
    MPI_Type_create_subarray(2, globalSize, localSize, starts, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, blockSize * sizeof(int), &subarrtype);
    MPI_Type_commit(&subarrtype);

    int *globalptrA = NULL;
    int *globalptrB = NULL;
    int *globalptrC = NULL;
    if (rank == 0) {
        globalptrA = &(matA[0][0]);
        globalptrB = &(matB[0][0]);
        globalptrC = &(resMat[0][0]);
    }

    // распределение данных по процессам
    int *sendCounts = (int *)malloc(sizeof(int) * countOfP);
    int *displacements = (int *)malloc(sizeof(int) * countOfP);

    if (rank == 0) {
        for (int i = 0; i < countOfP; i++) {
            sendCounts[i] = 1; // количества элементов для каждого процесса
        }
        int disp = 0;
        for (int i = 0; i < procGridDim; i++) {
            for (int j = 0; j < procGridDim; j++) {
                displacements[i * procGridDim + j] = disp;
                disp += 1;
            }
            disp += (blockSize - 1) * procGridDim;
        }
    }

    double time_start = MPI_Wtime();

    // Распределение данных из глобальных матриц по процессам
    MPI_Scatterv(globalptrA, sendCounts, displacements, subarrtype, &(localA[0][0]),
                 matrixSizeN * matrixSizeN / (countOfP), MPI_INT,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(globalptrB, sendCounts, displacements, subarrtype, &(localB[0][0]),
                 matrixSizeN * matrixSizeN / (countOfP), MPI_INT,
                 0, MPI_COMM_WORLD);

    int **localC = NULL;
    init_matrix(&localC, blockSize);
    // Инициализация результирующей матрицы
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            localC[i][j] = 0;
        }
    }
    // Определение соседей текущего процесса в решетке
    int coord[2]; // текущий процесс в решетке
    int left, right, up, down; // соседние процессы для обмена данными

    // Начальный сдвиг блоков
    MPI_Cart_coords(cartesianComm, rank, 2, coord);
    MPI_Cart_shift(cartesianComm, 1, coord[0], &left, &right);
    MPI_Sendrecv_replace(&(localA[0][0]), blockSize * blockSize, MPI_INT, left, 1, right, 1, cartesianComm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(cartesianComm, 0, coord[1], &up, &down);
    MPI_Sendrecv_replace(&(localB[0][0]), blockSize * blockSize, MPI_INT, up, 1, down, 1, cartesianComm, MPI_STATUS_IGNORE);

    int **multiplyRes = NULL;
    init_matrix(&multiplyRes, blockSize);
    for (int k = 0; k < procGridDim; k++) {
        // Умножение блоков matA и matB
        multiply_matrices(localA, localB, blockSize, &multiplyRes);

        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                localC[i][j] += multiplyRes[i][j];
            }
        }
        // Сдвиг блоков
        MPI_Cart_shift(cartesianComm, 1, 1, &left, &right);
        MPI_Cart_shift(cartesianComm, 0, 1, &up, &down);
        MPI_Sendrecv_replace(&(localA[0][0]), blockSize * blockSize, MPI_INT, left, 1, right, 1, cartesianComm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&(localB[0][0]), blockSize * blockSize, MPI_INT, up, 1, down, 1, cartesianComm, MPI_STATUS_IGNORE);
    }

    // Сбор результатов
    MPI_Gatherv(&(localC[0][0]), matrixSizeN * matrixSizeN / countOfP, MPI_INT,
                globalptrC, sendCounts, displacements, subarrtype,
                0, MPI_COMM_WORLD);


    double time_end = MPI_Wtime();

    if (rank == 0) {
        printf("Time: %f\n", time_end - time_start);
    }

    free_matrix(&localC);
    free_matrix(&multiplyRes);

    if (rank == 0) {
        // printf("Result:\n");
        // print_matrix_to_screen(resMat, matrixSizeN);
        // print_matrix_to_file(resMat, matrixSizeN, "output.txt");
    }

    MPI_Finalize();

    return 0;
}