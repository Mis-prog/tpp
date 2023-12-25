#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000 // Total threads
#define M 1024 // Threads per block

// Host - PC
// Device - GPU

void random_floats(float *x, int n, bool printRes);
void check_results(float *x, float *y, float *z, int n);

__global__ void add(float *a, float *b, float *c)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N) { c[index] = a[index] + b[index]; }
}

int main(void) 
{	
	srand(time(NULL));
	float *a = new float[N], *b = new float[N], *c = new float[N];
	float *a1, *b1, *c1;
	int size = sizeof(float) * N;
	bool printResults = false;
	
	printf("Size: %d\n", N);
	// Allocate space (without unified memory)
	cudaMalloc((void**)&a1, size);
	cudaMalloc((void**)&b1, size);
	cudaMalloc((void**)&c1, size);		

	// Initialize arrays
	random_floats(a, N, printResults);
	random_floats(b, N, printResults);
			
	cudaEvent_t startHTD, stopHTD;
	float timeHTD = 0;
	cudaEventCreate(&startHTD, 0);
	cudaEventCreate(&stopHTD, 0);
	cudaEventRecord(startHTD, 0);
	// copying data from host to device
	cudaMemcpy(a1, a, size, cudaMemcpyHostToDevice);	
	cudaMemcpy(b1, b, size, cudaMemcpyHostToDevice);	
//	cudaMemcpy(c1, c, size, cudaMemcpyHostToDevice);	
	cudaEventRecord(stopHTD, 0);
	cudaEventSynchronize(stopHTD);
	cudaEventElapsedTime(&timeHTD, startHTD, stopHTD);
	printf("Elapsed time from Host to Device: %.3f ms\n", timeHTD);
	cudaEventDestroy(startHTD);
	cudaEventDestroy(stopHTD);	
	
	// timings
	cudaEvent_t start, stop;
	float time = 0;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);
	cudaEventRecord(start, 0);

		
	add <<< (N + M - 1) / M, M>>> (a1, b1, c1); // Launch add() kernel on GPU
	
	// (N + M - 1 / M) - universal formula for the amount of blocks	
	// M - amount of threads
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed time : %.3f ms\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// timings
	
	cudaEvent_t startDTH, stopDTH;
	float timeDTH = 0;
	cudaEventCreate(&startDTH, 0);
	cudaEventCreate(&stopDTH, 0);
	cudaEventRecord(startDTH, 0);
	// copying data from device to host
//	cudaMemcpy(a, a1, size, cudaMemcpyDeviceToHost);	
//	cudaMemcpy(b, b1, size, cudaMemcpyDeviceToHost);	
	cudaMemcpy(c, c1, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stopDTH, 0);
	cudaEventSynchronize(stopDTH);
	cudaEventElapsedTime(&timeDTH, startDTH, stopDTH);
	printf("Elapsed time from Device to Host: %.3f ms\n", timeDTH);
	cudaEventDestroy(startDTH);
	cudaEventDestroy(stopDTH);
	
	cudaFree(a1);
	cudaFree(b1);
	cudaFree(c1);
	
	if (printResults)
	{
		check_results(a, b, c, N);
	}
	
	// Cleanup
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	delete[] a;
	delete[] b;
	delete[] c;
	return 0;
}

inline void check_cuda_errors(const char *filename, const int line_number) 
{
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) 
	{
		printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
		exit(-1);
	}
}

void random_floats(float *x, int n, bool printValues)
{
	for (int i = 0; i < n; i++)
	{
		x[i] = abs(sin(rand()) * 100);
		if (printValues)
		{
			printf("|sin(i)| * 100 = %f\n", x[i]);
		}
	}
} 

void check_results(float *x, float *y, float *z, int n)
{
	for (int i = 0; i < n; i++)
	{
		printf("%d", i);
		if (abs(x[i] + y[i] - z[i]) < 0.001)
		{
			printf("abs(%f + %f - %f) = %f\n", x[i], y[i], z[i], x[i]+y[i]-z[i]);
			printf("Valid result\n");
		}
		else
		{
			printf("abs(%f + %f - %f) != %f\n", x[i], y[i], z[i], x[i]+y[i]-z[i]);
			printf("Invalid result\n");
		}
	}
}
