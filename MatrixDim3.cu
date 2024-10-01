
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <chrono>


#define N 4  // Cambia a 4 para mostrar más fácilmente los resultados

// Kernel de CUDA para multiplicar matrices
__global__ void matrixMulCUDA(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int value = 0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

// Función para multiplicación de matrices en CPU
void matrixMulCPU(int* A, int* B, int* C, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            int value = 0;
            for (int k = 0; k < n; ++k) {
                value += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = value;
        }
    }
}

// Función para imprimir la matriz
void printMatrix(int* matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Reservar memoria para las matrices
    int* h_A, * h_B, * h_C_CPU, * h_C_GPU;
    int* d_A, * d_B, * d_C;

    size_t size = N * N * sizeof(int);
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C_CPU = (int*)malloc(size);
    h_C_GPU = (int*)malloc(size);

    // Inicializar las matrices A y B con valores aleatorios enteros
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() % 10;  // Números aleatorios entre 0 y 9
        h_B[i] = rand() % 10;  // Números aleatorios entre 0 y 9
    }

    // Medir tiempo en CPU
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_CPU, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCPU = endCPU - startCPU;
    std::cout << "Tiempo CPU: " << elapsedCPU.count() << " segundos" << std::endl;

    // Reservar memoria en GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copiar matrices de la CPU a la GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Definir dimensiones de los bloques y la cuadrícula
    dim3 threadsPerBlock(2, 2); // Ajustar para matrices pequeñas
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Medir tiempo en GPU
    auto startGPU = std::chrono::high_resolution_clock::now();
    matrixMulCUDA << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedGPU = endGPU - startGPU;
    std::cout << "Tiempo GPU: " << elapsedGPU.count() << " segundos" << std::endl;

    // Copiar resultado de la GPU a la CPU
    cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost);

    // Mostrar resultados
    std::cout << "Matriz A:" << std::endl;
    printMatrix(h_A, N);
    std::cout << "Matriz B:" << std::endl;
    printMatrix(h_B, N);
    std::cout << "Resultado CPU:" << std::endl;
    printMatrix(h_C_CPU, N);
    std::cout << "Resultado GPU:" << std::endl;
    printMatrix(h_C_GPU, N);
}