#include <stdio.h>
#include <tuple>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <windows.h>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

// For the CUDA runtime routines (prefixed with "cuda_")
using namespace std;

__global__ void tic_tac_toe(int *d_matrix, int *d_line, bool turn, int numElements);
__host__ tuple<int *, int *> allocateHostMemory(int size);
__host__ tuple<int *, int *> allocateDeviceMemory(int size);
__host__ void copyFromHostToDevice(int *h_matrix, int *h_line, int *d_matrix, int *d_line, int size);
__host__ void copyFromDeviceToHost(int *d_matrix, int &h_matrix, int size);
__host__ void deallocateMemory(int *h_matrix, int *d_matrix, int *h_line, int *d_line);
__host__ void cleanUpDevice();
__host__ void executeKernel(int *d_matrix, int *d_line, int numElements, int threadsPerBlock);
__host__ void displayBoard(int *matrix, int size);
__host__ void showAllCudaDevice();