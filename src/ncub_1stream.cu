/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

#include <iostream>

typedef unsigned int uint;

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 1
#endif

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}

	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
		printf("1: Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
		printf("1: Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

void print(uint* host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {
	uint num_of_segments;
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments + 1);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	int mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	uint *h_value = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_value[i] = i;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_vec, *d_vec_out;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));

	void *d_temp = NULL;
	size_t temp_bytes = 0;
	int num_of_streams = NUM_STREAMS;

	if(num_of_streams > num_of_segments){
		num_of_streams = num_of_segments;
	}

	cudaStream_t streams[NUM_STREAMS];
	for(int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	for (uint e = 0; e < EXECUTIONS; e++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

		cudaEventRecord(start);
		for (int i = 0; i < num_of_segments; i+=num_of_streams) {
			for (int s = 0; s < num_of_streams; s++) {
				cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_vec+h_seg[i+s], d_vec_out+h_seg[i+s],
						h_seg[i+1+s]-h_seg[i+s], 0, sizeof(uint)*8, streams[s]);
				cudaMalloc((void **) &d_temp, temp_bytes);
				cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_vec+h_seg[i+s], d_vec_out+h_seg[i+s],
						h_seg[i+1+s]-h_seg[i+s], 0, sizeof(uint)*8, streams[s]);
			}
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("4: Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("4: Async kernel error: %s\n", cudaGetErrorString(errAsync));

		if (ELAPSED_TIME == 1) {
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();

		cudaFree(d_temp);
		temp_bytes = 0;
		d_temp = NULL;
	}

	cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost);

	cudaFree(streams);
	cudaFree(d_vec);
	cudaFree(d_vec_out);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_elements);
	}

	free(h_seg);
	free(h_vec);
	free(h_value);

	return 0;
}
