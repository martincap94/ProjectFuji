
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include <iostream>


using namespace std;

int main() {



	int n = 9;

	// 9 elements, last index = 8
	int arr[] = { 0, 4, 3, 2, 0, 5, 2, 8, 0 }; 
	int sums[] = { 0, 4, 7, 9, 9, 14, 16, 24, 24 };


	//int arr[] = { 9, 0, 0, 0, 0, 0, 0, 0, 0 };
	//int sums[] = { 9, 9, 9, 9, 9, 9, 9, 9, 9 };


	//int *sums = new int[n]();

	



	// binary search test

	int left = 0;
	int right = n - 1;

	int val = 24; // "randomly" generated value

	int idx;
	while (left <= right) {
		cout << "left = " << left << ", right = " << right << endl;
		idx = (left + right) / 2;
		cout << "idx = " << idx << endl;
		if (val <= sums[idx]) {
			right = idx - 1;
		} else {
			left = idx + 1;
		}
	}
	idx = left;


	cout << "idx = " << idx << " (val = " << sums[idx] << ", probability = " << arr[idx] << ")" << endl;


	thrust::device_vector<int> input(n);
	for (int i = 0; i < n; i++) {
		input[i] = sums[i];
	}
	cout << "THRUST:" << endl;
	idx = thrust::distance(input.begin(), thrust::lower_bound(input.begin(), input.end(), val));
	cout << "idx = " << idx << " (val = " << sums[idx] << ", probability = " << arr[idx] << ")" << endl;



}
