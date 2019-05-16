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
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

//#include <future>
#include <thread>
#include <chrono>
#include <iostream>
#include <omp.h>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 4
#endif

void print(thrust::host_vector<int> h_vec) {
	std::cout << "\n";
	for (int i = 0; i < h_vec.size(); i++) {
		std::cout << h_vec[i] << " ";
	}
	std::cout << "\n";
}

template<typename D, typename H>
void kernelCall(cudaStream_t stream, D d_vec, H h_seg, int i){

	cudaSetDevice(0);
	printf("teste segment %d\n", i);
	thrust::sort(thrust::cuda::par.on(stream),d_vec.begin() + h_seg[i], d_vec.begin() + h_seg[i + 1]);


}

int main(void) {
	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	thrust::host_vector<int> h_seg(num_of_segments + 1);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	thrust::host_vector<int> h_vec(num_of_elements);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	thrust::device_vector<uint> d_vec(num_of_elements);

	cudaStream_t streams[NUM_STREAMS];
	for(int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
		//cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
	}

	int nstreams = NUM_STREAMS;
	if(NUM_STREAMS > num_of_segments)
		nstreams = num_of_segments;

	omp_set_num_threads(nstreams);

	for (uint i = 0; i < EXECUTIONS; i++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

		cudaEventRecord(start);
		#pragma omp parallel
		{
			uint id = omp_get_thread_num(); //cpu_thread_id
			for (int i = 0; i < num_of_segments; i+=nstreams) {
				uint k = i + id;
				thrust::sort(thrust::cuda::par.on(streams[id]), d_vec.begin() + h_seg[k], d_vec.begin() + h_seg[k + 1]);
			}
		}
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

	if (ELAPSED_TIME != 1) {
		print(h_vec);
	}

	cudaFree(streams);

	return 0;
}


/*thrust::sort(thrust::cuda::par.on(streams[0]), d_vec.begin() + h_seg[i],
		d_vec.begin() + h_seg[i + 1]);
thrust::sort(thrust::cuda::par.on(streams[1]), d_vec.begin() + h_seg[i+1],
		d_vec.begin() + h_seg[i + 2]);

thrust::sort(thrust::cuda::par.on(streams[2]), d_vec.begin() + h_seg[i+2],
		d_vec.begin() + h_seg[i + 3]);*/
//thrust::sort(thrust::cuda::par.on(streams[3]), d_vec.begin() + h_seg[i+3],d_vec.begin() + h_seg[i + 4]);
//std::async(std::launch::async, &kernelCall,streams[0], d_vec, h_seg, i);
//std::thread t1(&kernelCall<thrust::device_vector<uint>, thrust::host_vector<int>>,streams[0], d_vec, h_seg, i+0);
//std::thread t2(&kernelCall<thrust::device_vector<uint>, thrust::host_vector<int>>,streams[1], d_vec, h_seg, i+1);
//std::thread t3(&kernelCall<thrust::device_vector<uint>, thrust::host_vector<int>>,streams[2], d_vec, h_seg, i+2);
//std::thread t4(&kernelCall<thrust::device_vector<uint>, thrust::host_vector<int>>,streams[3], d_vec, h_seg, i+3);
//t1.join();
//t2.join();
//t3.join();
//t4.join();

//std::async(kernelCall,streams[1], d_vec, h_seg, i+1);
//std::async(kernelCall,streams[2], d_vec, h_seg, i+2);
//std::async(kernelCall,streams[3], d_vec, h_seg, i+3);
