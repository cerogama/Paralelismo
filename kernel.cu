
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void cuda_hello()
{
	int n = 0;
	printf("hello world from GPU \n");

}

__global__ void add_Gpu(int* a, int* b, int* c)
{
	printf("Add with Gpu: \n");
	*c = *a + *b;
}

__global__ void substraction_Gpu(int* a, int* b, int* c)
{
	printf("Substraction with Gpu: \n");
	*c = *a - *b;
}

__global__ void multiply_Gpu(int* a, int* b, int* c)
{
	printf("Multiply with Gpu: \n");
	*c = *a * *b;
}

__global__ void Division_Gpu(int* a, int* b, int* c)
{
	printf("Division with Gpu: \n");
	*c = *a / *b;
}

void add_Cpu(int* a, int* b, int* c)
{
	printf("Add with Cpu: \n");
	*c = *a + *b;
}

void substraction_Cpu(int* a, int* b, int* c)
{
	printf("Substraction with Cpu: \n");
	*c = *a - *b;
}

void multiply_Cpu(int* a, int* b, int* c)
{
	printf("Multiply with Cpu: \n");
	*c = *a * *b;
}

void division_Cpu(int* a, int* b, int* c)
{
	printf("Division with Cpu: \n");
	*c = *a / *b;
}

int main()
{
	int a = 2;
	int* b;
	int c;
	b = &c;
	*b = 10;
	c = 10;
	int result, gpu_Result_1, gpu_Result_2, gpu_Result_3;

	int cpu_Result, cpu_Result_1, cpu_Result_2, cpu_Result_3;

	int* gpu_a, * gpu_b, * gpu_result;

	int size = sizeof(int);
	int size_a = sizeof(a);

	cudaMalloc((void**)&gpu_a, size);
	cudaMalloc((void**)&gpu_b, size);
	cudaMalloc((void**)&gpu_result, size);

	cudaMemcpy(gpu_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, &c, size, cudaMemcpyHostToDevice);
	
	substraction_Gpu << < 1, 1 >> > (gpu_a, gpu_b, gpu_result);
	cudaMemcpy(&result, gpu_result, size, cudaMemcpyDeviceToHost);
	printf("Result gpu: %d\n", result);
	
	substraction_Gpu << < 1, 1 >> > (gpu_a, gpu_b, gpu_result);
	cudaMemcpy(&gpu_Result_2, gpu_result, size, cudaMemcpyDeviceToHost);
	printf("Result gpu: %d\n", gpu_Result_2);

	multiply_Gpu << < 1, 1 >> > (gpu_a, gpu_b, gpu_result);
	cudaMemcpy(&gpu_Result_1, gpu_result, size, cudaMemcpyDeviceToHost);
	printf("Result gpu: %d\n", gpu_Result_1);

	Division_Gpu << < 1, 1 >> > (gpu_a, gpu_b, gpu_result);
	cudaMemcpy(&gpu_Result_3, gpu_result, size, cudaMemcpyDeviceToHost);
	printf("Result gpu: %d\n", gpu_Result_3);

	//----------------------------
	//			CPU

	add_Cpu(&a, b, &cpu_Result);
	printf("Result: %d\n", cpu_Result);

	substraction_Cpu(&a, b, &cpu_Result_1);
	printf("Result: %d\n", cpu_Result_1);

	multiply_Cpu(&a, b, &cpu_Result_2);
	printf("Result: %d\n", cpu_Result_2);

	division_Cpu(&a, b, &cpu_Result_3);
	printf("Result: %d\n", cpu_Result_3);
	
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_result);
	return 0;
}

