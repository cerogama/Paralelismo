#include <iostream>
#include <cuda_runtime.h>
#include <chrono> // Para medir tiempos
#include "bitmap_image.hpp" // Biblioteca para manejar imágenes BMP

// Clase que representa números complejos
class cuComplex {
public:
    float r, i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ cuComplex operator*(const cuComplex& a) const {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a) const {
        return cuComplex(r + a.r, i + a.i);
    }

    __device__ float magnitude2() const {
        return r * r + i * i;
    }
};

// Función que calcula si un punto pertenece al conjunto de Julia
__device__ int julia(int x, int y, int n) {
    const float scale = 1.5;
    float jx = scale * (float)(n / 2 - x) / (n / 2.f);
    float jy = scale * (float)(n / 2 - y) / (n / 2.f);
    cuComplex c(0.064, 0.626);//valores constantes de la funcion julia

    cuComplex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) {
            return 0; // El punto está en el conjunto
        }
    }
    return 1; // El punto no está en el conjunto
}

// Kernel de CUDA que calcula el conjunto de Julia
__global__ void computeJulia(unsigned char* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = (y * width + x) * 3;
        int value = julia(x, y, width);

        if (value) {
            pixels[index + 0] = 255; // Rojo
            pixels[index + 1] = 0;   // Verde
            pixels[index + 2] = 0;   // Azul
        }
        else {
            pixels[index + 0] = 0;   // Negro
            pixels[index + 1] = 0;
            pixels[index + 2] = 0;
        }
    }
}

// Función para guardar la imagen generada
void saveImage(const unsigned char* pixels, int width, int height) {
    bitmap_image image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * 3;
            unsigned char r = pixels[index + 0];
            unsigned char g = pixels[index + 1];
            unsigned char b = pixels[index + 2];
            image.set_pixel(x, y, r, g, b);
        }
    }
    image.save_image("julia_set.bmp");
}

int main() {
    const int width = 5000;
    const int height = 5000;

    unsigned char* pixels;
    unsigned char* d_pixels;

    // Reservar memoria para los pixels en el host
    pixels = new unsigned char[width * height * 3];
    cudaMalloc(&d_pixels, width * height * 3 * sizeof(unsigned char));

    // Configurar el tamaño de bloque y de la cuadrícula
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Medir el tiempo del kernel
    auto startKernel = std::chrono::high_resolution_clock::now();
    computeJulia << <gridSize, blockSize >> > (d_pixels, width, height);
    cudaDeviceSynchronize();
    auto endKernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedKernel = endKernel - startKernel;

    // Medir el tiempo de la copia de memoria
    auto startCopy = std::chrono::high_resolution_clock::now();
    cudaMemcpy(pixels, d_pixels, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    auto endCopy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCopy = endCopy - startCopy;

    // Guardar la imagen generada
    saveImage(pixels, width, height);

    // Liberar memoria
    delete[] pixels;
    cudaFree(d_pixels);

    // Imprimir tiempos
    std::cout << "Tiempo de ejecución del kernel: " << elapsedKernel.count() << " segundos." << std::endl;
    std::cout << "Tiempo de copia de memoria: " << elapsedCopy.count() << " segundos." << std::endl;
    std::cout << "Imagen del conjunto de Julia generada y guardada como 'julia_set.bmp'." << std::endl;

    return 0;
}
