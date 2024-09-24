#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void multiplicationMatrix_GPU(int* matrix1, int* matrix2, int* multiplication, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < rows * cols) {
        multiplication[idx] = matrix1[idx] * matrix2[idx];
    }
}

void multiplicationMatrix_CPU(int* matrix1, int* matrix2, int* multiplication, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            multiplication[i * cols + j] = matrix1[i * cols + j] * matrix2[i * cols + j];
        }
    }
}

void showMatrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int N = 20 * (1 << 20);
    int rows, cols;

    std::cout << "Ingrese el número de filas: ";
    std::cin >> rows;
    std::cout << "Ingrese el número de columnas: ";
    std::cin >> cols;

    // Asignar memoria para matrices en la CPU
    int* h_matrix1 = new int[rows * cols];
    int* h_matrix2 = new int[rows * cols];
    int* h_multiplication = new int[rows * cols];

    // Leer valores de la matriz
    std::cout << "Ingrese los valores de la primera matriz:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << "Element[" << i << "][" << j << "]: ";
            std::cin >> h_matrix1[i * cols + j];
        }
    }

    std::cout << "Ingrese los valores de la segunda matriz:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << "Element[" << i << "][" << j << "]: ";
            std::cin >> h_matrix2[i * cols + j];
        }
    }

    // Asignar memoria para matrices en la GPU
    int* d_matrix1, * d_matrix2, * d_multiplication;
    cudaMalloc((void**)&d_matrix1, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_matrix2, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_multiplication, rows * cols * sizeof(int));

    //Creacion de eventos para medir el tiempo de ejecucion
    cudaEvent_t startGpu, stopGpu, startCpu, stopCpu;
    cudaEventCreate(&startGpu);
    cudaEventCreate(&stopGpu);
    cudaEventCreate(&startCpu);
    cudaEventCreate(&stopCpu);

    // Copiar matrices desde la CPU a la GPU
    cudaMemcpy(d_matrix1, h_matrix1, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2, rows * cols * sizeof(int), cudaMemcpyHostToDevice);


    cudaEventRecord(startGpu);
    // Llamar al kernel sin definir blockSize y gridSize
    multiplicationMatrix_GPU << <(rows * cols + 255) / 256, 256 >> > (d_matrix1, d_matrix2, d_multiplication, rows, cols);
    /*
        la funcion: cudaDeviceSynchronize() pausa momentaneamente el host hasta queel kernel acabe, verifica errores(argumentar mejor),
        se asegura que los kernels funcionen uno tras otro, se asegura que operaciones copia sean complementarias antes de continuar con otras operaciones en el host
    */
    cudaEventRecord(stopGpu);

    cudaDeviceSynchronize();

    // Copiar el resultado de vuelta a la CPU
    cudaMemcpy(h_multiplication, d_multiplication, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
  
    cudaEventRecord(startCpu);
    // Multiplicar matrices en CPU
    multiplicationMatrix_CPU(h_matrix1, h_matrix2, h_multiplication, rows, cols);
    cudaEventRecord(stopCpu);


    cudaEventSynchronize(stopGpu);
    float millisecondsGPU = 0;
    cudaEventElapsedTime(&millisecondsGPU, startGpu, stopGpu);

    cudaEventSynchronize(stopCpu);
    float millisecondsCPU = 0;
    cudaEventElapsedTime(&millisecondsCPU, startCpu, stopCpu);

    // Muestra la matriz resultante de la GPU
    std::cout << "Matriz resultante de la multiplicacion en GPU:\n";
    showMatrix(h_multiplication, rows, cols);
   

    // Muestra la matriz resultante de la CPU
    std::cout << "Matriz resultante de la multiplicacion en CPU:\n";
    showMatrix(h_multiplication, rows, cols);
    
    printf("Tiempo de espera en GPU (GB/s): %fn", N * 4 * 3 / millisecondsGPU / 1e6);
    std::cout << std::endl;
    printf("Tiempo de espera en CPU (GB/s): %fn", N * 4 * 3 / millisecondsCPU / 1e6);
    std::cout << std::endl;

    // Liberar memoria
    delete[] h_matrix1;
    delete[] h_matrix2;
    delete[] h_multiplication;
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_multiplication);

    return 0;
}
