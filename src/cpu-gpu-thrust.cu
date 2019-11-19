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
#include <thrust/system/cuda/detail/par.h>

//#include <future>
//#include <thread>

#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

#include <cudaProfiler.h>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 32
#endif

void print(uint* h_vec, int n) {
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << h_vec[i] << " ";
	}
	std::cout << "\n";
}

//template<class T>
void kernelCall(thrust::system::cuda::detail::execute_on_stream exec, thrust::detail::normal_iterator<thrust::device_ptr<uint>> first, thrust::detail::normal_iterator<thrust::device_ptr<uint>> last){
//void kernelCall(thrust::cuda_cub::execute_on_stream exec, thrust::detail::normal_iterator<thrust::device_ptr<uint>> first, thrust::detail::normal_iterator<thrust::device_ptr<uint>> last){
	thrust::sort(exec,first,last);
}

int main(void) {
	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments + 1);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec_aux = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec_aux[i]);

	cudaStream_t streams[NUM_STREAMS];
	for(int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
		//cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
	}

	uint* d_vec;
	cudaMallocManaged((void **)&d_vec, sizeof(uint)*num_of_elements);

	int nstreams = NUM_STREAMS;
	if(NUM_STREAMS > num_of_segments)
		nstreams = num_of_segments;

	omp_lock_t semaphore_lock;
	omp_init_lock(&semaphore_lock);
	
	for (uint i = 0; i < EXECUTIONS; i++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		for (i = 0; i < num_of_elements; i++)
			d_vec[i] = h_vec_aux[i];

		cuProfilerStart();
		cudaEventRecord(start);

		omp_set_num_threads(nstreams);
		int s = 0;
		#pragma omp parallel
		{
			uint id = omp_get_thread_num(); //cpu_thread_id

			if(id < 0){
				while(true) {
					omp_set_lock(&semaphore_lock);
					uint k = s;
					s++;
					omp_unset_lock(&semaphore_lock);

					if(k >= num_of_segments) {
						break;
					}
					//printf("i=%d   ---   k=%d\n", s, k);

					//thrust::sort(thrust::cuda::par.on(streams[id]), d_vec.begin() + h_seg[k], d_vec.begin() + h_seg[k + 1]);
					thrust::sort(thrust::cuda::par.on(streams[id]), d_vec + h_seg[k], d_vec + h_seg[k + 1]);
				}
			}
			else {
				while(true) {
					omp_set_lock(&semaphore_lock);
					uint k = s;
					s++;
					omp_unset_lock(&semaphore_lock);

					if(k >= num_of_segments) {
						break;
					}
					//printf("i=%d   ---   k=%d\n", s, k);

					std::stable_sort(&d_vec[h_seg[k]], &d_vec[h_seg[i+1]]);
				}
			}
		}

		cudaEventRecord(stop);
		cuProfilerStop();

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	if (ELAPSED_TIME != 1) {
		print(d_vec, num_of_elements);
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
