#include<stdio.h>
#include<stdlib.h>
#include <iostream>
#include <vector>

using namespace std;

int findMaxInBitonicArray(const vector<int>& arr) {
    int low = 0;
    int high = arr.size() - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if ((mid == 0 || arr[mid] > arr[mid - 1]) && 
            (mid == arr.size() - 1 || arr[mid] > arr[mid + 1])) {
            return arr[mid];
        }

        if (arr[mid] < arr[mid + 1]) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return -1; 
}

int main() {
    vector<int> arr = {1, 2, 3, 10, 9, 8, 7};
    int max = findMaxInBitonicArray(arr);

    if (max != -1) {
        cout << "The maximum element in the bitonic array is: " << max << endl;
    } else {
        cout << "Invalid input array." << endl;
    }

    return 0;
}