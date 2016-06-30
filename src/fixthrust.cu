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
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <chrono>
#include <iostream>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

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
	uint i;

	scanf("%d", &num_of_segments);
	thrust::host_vector<uint> h_seg(num_of_segments + 1);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	thrust::host_vector<uint> h_vec(num_of_elements);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	uint maxValue = 0;
	for (i = 0; i < num_of_elements; i++) {
		if(maxValue < h_vec[i])
			maxValue = h_vec[i];
	}

	uint mostSignificantBit = (uint)log2((double)maxValue) + 1;

	for (i = 0; i < num_of_segments; i++) {
		for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
			uint segIndex = i << mostSignificantBit;
			h_vec[j] += segIndex;
		}
	}

	thrust::device_vector<uint> d_vec(num_of_elements);

	for (uint i = 0; i < EXECUTIONS; i++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

		cudaEventRecord(start);
		thrust::sort(d_vec.begin(), d_vec.end());
		cudaEventRecord(stop);

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	for (i = 0; i < num_of_segments; i++) {
		for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
			uint segIndex = i << mostSignificantBit;
			h_vec[j] -= segIndex;
		}
	}

	if (ELAPSED_TIME != 1) {
		print(h_vec);
	}

	return 0;
}
