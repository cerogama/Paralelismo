
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

void printDeviceProperties(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, deviceId);

    if (err != cudaSuccess) {
        std::cerr << "Error retrieving device properties: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    //Nombre y modelo de GPU
    std::cout << "Device Name: " << deviceProp.name << std::endl;
    
    //Cantidad total de memoria de la GPU
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
   
    //Memori rapida que puede ser compartida entre hilos
    std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    
    //Numero maximo de hilos que puede ser soportado
    std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;

    //Las dimensiones que puede tener un grupo de hilos
    std::cout << "Max Block Dimensions: [" << deviceProp.maxThreadsDim[0] << ", "
        << deviceProp.maxThreadsDim[1] << ", "
        << deviceProp.maxThreadsDim[2] << "]" << std::endl;
    
    //El tamaño maximo que puede tener la maya que puede dividirse en varios grupos de hilos
    std::cout << "Max Grid Dimensions: [" << deviceProp.maxGridSize[0] << ", "
        << deviceProp.maxGridSize[1] << ", "
        << deviceProp.maxGridSize[2] << "]" << std::endl;

    //Un número que indica la potencia y las características de la GPU
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

    //Capacidad de calculo de la GPU
    std::cout << "Clock Rate: " << deviceProp.clockRate << " kHz" << std::endl;
    
    //La velocidad a la que la memoria de la GPU puede transferir datos
    std::cout << "Memory Clock Rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;
    
    //Cuántas partes de la GPU pueden trabajar al mismo tiempo
    std::cout << "Number of Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        std::cout << "Device " << i << " properties:" << std::endl;
        printDeviceProperties(i);
        std::cout << std::endl;
    }

    return 0;
}
