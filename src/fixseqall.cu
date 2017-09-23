/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */


#include <stdio.h>
#include <stdlib.h>
#include <algorithm>    // std::sort
#include <iostream>
#include <cuda.h>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

template<typename T>
struct Element {
	int segment;
	T value;

	Element(int segment, T value){
		this->segment = segment;
		this->value = value;
	}

	Element(){
		this->segment = 0;
	}
};

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
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec_aux = (Element *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		for(int j = h_seg[i]; j < h_seg[i+1]; j++) {
			scanf("%d", &h_vec_aux[i]);
		}
	}

	uint *h_vec = (uint *) malloc(mem_size_vec);
	uint mostSignificantBit = 0;
	for (uint k = 0; k < EXECUTIONS; k++) {

		for (i = 0; i < num_of_elements; i++) {
			h_vec[i] = h_vec_aux[i];
		}

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		uint maxValue = 0;
		for (i = 0; i < num_of_elements; i++) {
			if(maxValue < h_vec[i])
				maxValue = h_vec[i];
		}

		mostSignificantBit = (uint)log2((double)maxValue) + 1;

		for (i = 0; i < num_of_segments; i++) {
			for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
				uint segIndex = i << mostSignificantBit;
				h_vec[j] += segIndex;
			}
		}

		std::stable_sort(&h_vec[0], &h_vec[num_of_elements]);

		for (i = 0; i < num_of_segments; i++) {
			for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
				uint segIndex = i << mostSignificantBit;
				h_vec[j] -= segIndex;
			}
		}
		cudaEventRecord(stop);

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

	}

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_elements);
	}

	free(h_seg);
	free(h_vec);
	free(h_vec_aux);

	return 0;
}
