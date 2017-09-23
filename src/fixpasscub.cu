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


template <typename T>
struct Plus {
	__host__ __device__
	T operator()(const T x, const T y)
	{
		return x + y;
	}
};


template <typename T>
struct Minus {
	__host__ __device__
	T operator()(const T x, const T y)
	{
		return x - y;
	}
};

template<typename Op>
__global__ void adjustment(uint* d_vec, uint* d_seg, uint num_of_elements, uint* d_max ){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < num_of_elements) {
		uint mostSignificantBit = (uint)log2((double)*d_max) + 1;
		uint segIndex = d_seg[id] << mostSignificantBit;
		Op op = Op();
		d_vec[id] = op(d_vec[id], segIndex);
	}
}


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
	uint *h_seg_aux = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg_aux[i]);

	scanf("%d", &num_of_elements);
	int mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	uint *h_value = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_value[i] = i;
	}

	uint *h_seg = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_segments; i++) {
		for (uint j = h_seg_aux[i]; j < h_seg_aux[i + 1]; j++) {
			h_seg[j] = i;
		}
	}

	cudaEvent_t startPre, stopPre, startPos, stopPos;
	cudaEventCreate(&startPre);
	cudaEventCreate(&stopPre);
	cudaEventCreate(&startPos);
	cudaEventCreate(&stopPos);

	uint *d_value, *d_value_out, *d_vec, *d_vec_out, *d_max, *d_seg;
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	uint* max_val = (uint *) malloc(sizeof(uint));

	cudaTest(cudaMalloc((void **) &d_max, sizeof(uint)));
	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_seg, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

	cudaTest(cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_vec, cudaMemcpyHostToDevice));

	void *d_temp = NULL;
	size_t temp_bytes = 0;
	int grid = ((num_of_elements-1)/BLOCK_SIZE) + 1;

	for (uint i = 0; i < EXECUTIONS; i++) {
		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

		/*
		 * maximum element of the array.
		 */
		cudaEventRecord(startPre);
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_vec, d_max, num_of_elements);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);	// Allocate temporary storage
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_vec, d_max, num_of_elements);	// Run max-reduction

		/*
		 * add prefix to the elements
		 */
		adjustment<Plus<uint>> <<< grid, BLOCK_SIZE>>>(d_vec, d_seg, num_of_elements, d_max);
		cudaEventRecord(stopPre);
		cudaEventSynchronize(stopPre);

		/*
		 * sort the vector
		 */
		cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out,
				d_value, d_value_out, num_of_elements);
		cudaMalloc((void **) &d_temp, temp_bytes);
		cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out,
				d_value, d_value_out, num_of_elements);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("4: Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("4: Async kernel error: %s\n", cudaGetErrorString(errAsync));

		cudaEventRecord(startPos);
		adjustment<Minus<uint>> <<< grid, BLOCK_SIZE>>>(d_vec_out, d_seg, num_of_elements, d_max);
		cudaEventRecord(stopPos);
		cudaEventSynchronize(stopPos);

		if (ELAPSED_TIME == 1) {
			float millisecondsPre = 0, millisecondsPos = 0;
			cudaEventElapsedTime(&millisecondsPre, startPre, stopPre);
			cudaEventElapsedTime(&millisecondsPos, startPos, stopPos);
			std::cout << millisecondsPre + millisecondsPos << "\n";
		}

		cudaFree(d_temp_storage);
		temp_storage_bytes = 0;
		d_temp_storage = NULL;

		cudaFree(d_temp);
		temp_bytes = 0;
		d_temp = NULL;

		cudaDeviceSynchronize();
	}

	cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost);

	cudaFree(d_max);
	cudaFree(d_seg);
	cudaFree(d_vec);
	cudaFree(d_vec_out);
	cudaFree(d_value);
	cudaFree(d_value_out);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_elements);
	}

	free(h_seg_aux);
	free(h_seg);
	free(h_vec);
	free(h_value);

	return 0;
}
