/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <iostream>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

template <typename T, typename Op>
struct Operation {

	uint shift_val;

	Operation(uint shift_val) {
		this->shift_val = shift_val;
	}

	__host__ __device__
	T operator()(const T x, const T y)
	{
		T fix = y << shift_val;
		Op op = Op();
		return op(x, fix);
	}
};

void print(thrust::host_vector<uint> h_vec) {
	std::cout << "\n";
	for (uint i = 0; i < h_vec.size(); i++) {
		std::cout << h_vec[i] << " ";
	}
	std::cout << "\n";
}

int main(void) {
	uint num_of_segments;
	uint num_of_elements;

	scanf("%d", &num_of_segments);
	thrust::host_vector<uint> h_seg_aux(num_of_segments + 1);
	for (uint i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg_aux[i]);

	scanf("%d", &num_of_elements);
	thrust::host_vector<uint> h_vec(num_of_elements);
	for (uint i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	thrust::host_vector<uint> h_seg(num_of_elements);
	for (uint i = 0; i < num_of_segments; i++) {
		for (uint j = h_seg_aux[i]; j < h_seg_aux[i + 1]; j++) {
			h_seg[j] = i;
		}
	}

	cudaEvent_t startPre, stopPre, startPos, stopPos;
	cudaEventCreate(&startPre);
	cudaEventCreate(&stopPre);
	cudaEventCreate(&startPos);
	cudaEventCreate(&stopPos);

	thrust::device_vector<uint> d_vec(num_of_elements);
	thrust::device_vector<uint> d_seg = h_seg;

	for (uint i = 0; i < EXECUTIONS; i++) {
		thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
		/*
		 * maximum element of the array.
		 */
		cudaEventRecord(startPre);
		thrust::device_vector<uint>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());
		uint max_val = *iter;
		uint mostSignificantBit = (uint)log2((double)max_val) + 1;
		/*
		 * add prefix to the elements
		 */
		Operation< uint, thrust::plus<uint> > op_plus(mostSignificantBit);
		thrust::transform(d_vec.begin(), d_vec.end(), d_seg.begin(), d_vec.begin(), op_plus);
		cudaEventRecord(stopPre);
		cudaEventSynchronize(stopPre);
		/*
		 * sort the segments
		 */
		thrust::sort(d_vec.begin(), d_vec.end());
		/*
		 * update back the array elements
		 */
		cudaEventRecord(startPos);
		Operation< uint, thrust::minus<uint> > op_minus(mostSignificantBit);
		thrust::transform(d_vec.begin(), d_vec.end(), d_seg.begin(), d_vec.begin(), op_minus);
		cudaEventRecord(stopPos);
		cudaEventSynchronize(stopPos);

		if (ELAPSED_TIME == 1) {
			float millisecondsPre = 0, millisecondsPos = 0;
			cudaEventElapsedTime(&millisecondsPre, startPre, stopPre);
			cudaEventElapsedTime(&millisecondsPos, startPos, stopPos);
			std::cout << millisecondsPre + millisecondsPos << "\n";
		}

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("4: Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("4: Async kernel error: %s\n", cudaGetErrorString(errAsync));
	}

	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	if (ELAPSED_TIME != 1) {
		print(h_vec);
	}

	return 0;
}
