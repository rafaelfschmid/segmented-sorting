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

#include <future>
#include <thread>

#include <iostream>
#include <omp.h>
#include <vector>

#include <cudaProfiler.h>

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

//template<class T>
//void kernelCall(thrust::system::cuda::detail::execute_on_stream exec, thrust::detail::normal_iterator<thrust::device_ptr<uint>> first, thrust::detail::normal_iterator<thrust::device_ptr<uint>> last){
void kernelCall(thrust::cuda_cub::execute_on_stream exec, thrust::detail::normal_iterator<thrust::device_ptr<uint>> first, thrust::detail::normal_iterator<thrust::device_ptr<uint>> last){
	thrust::sort(exec,first,last);
}

int main(void) {
	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	thrust::host_vector<int> h_seg(num_of_segments + 1);
	for (i = 0; i < num_of_segments + 1; i++) {
		scanf("%d", &h_seg[i]);
	}


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


	float averageExecutions = 0;
	for (uint i = 0; i < EXECUTIONS; i++) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

		cuProfilerStart();
		cudaEventRecord(start);

		#ifdef SCHEDULE
		std::vector<uint> amountOfwork(nstreams); //vetor para escalonar os kernels
		std::vector<uint> segMachine(num_of_segments);

		for(int j = 0; j < nstreams; j++) {
			amountOfwork[j] = 0;
		}

		for(int i = 0; i < num_of_segments; i++) {
			int min = 0;
			for(int j = 1; j < nstreams; j++) {
				if(amountOfwork[j] < amountOfwork[min])
					min = j;
			}
			amountOfwork[min] += h_seg[i+1] - h_seg[i];
			segMachine[i] = min;
		}
		#endif

		/*
		 * async with scheduling
		 */
		#ifdef SCHEDULE

			#ifdef THREADS
				omp_set_num_threads(nstreams);
				#pragma omp parallel
				{
					uint id = omp_get_thread_num(); //cpu_thread_id

					for (int i = 0; i < num_of_segments; i+=nstreams) {
						uint k = i + id;
						thrust::sort(thrust::cuda::par.on(streams[segMachine[k]]), d_vec.begin() + h_seg[k], d_vec.begin() + h_seg[k + 1]);
					}
				}
            #else

				std::vector<std::future<void>> vec;
				for (int k = 0; k < num_of_segments; k++) {
					vec.push_back(std::async(std::launch::async, kernelCall, thrust::cuda::par.on(streams[segMachine[k]]), d_vec.begin() + h_seg[k], d_vec.begin() + h_seg[k + 1]));
				}
				for(int k = 0; k < vec.size(); k++){
					vec[k].get();
				}
			#endif
		#else
			#ifdef THREADS
				omp_set_num_threads(nstreams);
				#pragma omp parallel
				{
					uint id = omp_get_thread_num(); //cpu_thread_id
					for (int i = 0; i < num_of_segments; i+=nstreams) {
						uint k = i + id;
						thrust::sort(thrust::cuda::par.on(streams[id]), d_vec.begin() + h_seg[k], d_vec.begin() + h_seg[k + 1]);
					}
				}
			#else

				std::vector<std::future<void>> vec;
				int id = 0;
				for (int i = 0; i < num_of_segments; i++) {
					vec.push_back(std::async(std::launch::async, kernelCall, thrust::cuda::par.on(streams[id]), d_vec.begin() + h_seg[i], d_vec.begin() + h_seg[i + 1]));
					id = (id+1) % nstreams;
				}
				for(int k = 0; k < vec.size(); k++){
					vec[k].get();
				}

				/*
				 * async withOUT scheduling
				 */
				/*for (int id = 0; id < machines.size(); id++) {
					for (int i = 0; i < machines[id].size(); i++) {
						thrust::sort(thrust::cuda::par.on(streams[id]), d_vec.begin() + h_seg[machines[id][i]], d_vec.begin() + h_seg[machines[id][i] + 1]);
					}
				}*/
			#endif

		#endif

		cudaEventRecord(stop);
		cuProfilerStop();

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			averageExecutions += milliseconds;
		}

		cudaDeviceSynchronize();
	}

	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	if (ELAPSED_TIME != 1) {
		print(h_vec);
	}
	else {
			std::cout << averageExecutions/EXECUTIONS << "\n";
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
