#include "tic-tac-toe.h"

__global__ void tic_tac_toe(int *d_matrix, int *d_line, int numElements, int currentPlayer)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < 1)
    {
        int priority[9] = {4, 0, 2, 6, 8, 1, 3, 5, 7};
        // 3 4 5
        // 0 1 2 0
        // 3 4 5 1 
        // 6 7 8 2
        int graph[9][4] = {
            {0, 3, 6}, 
            {0, 4}, 
            {0, 5, 7}, 
            {1, 3}, 
            {1, 4, 6, 7}, 
            {1, 5}, 
            {2, 3, 7}, 
            {2, 4}, 
            {2, 5, 6}
        };

        // search for win
        int attack = -1;
        int defence = -1;
        int chosen = -1;
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(d_matrix[i] == -1 && abs(d_line[graph[i][j]]) == 2){
                    if(currentPlayer == 1){
                        if(d_line[graph[i][j]] > 0){
                            attack = i;
                            break;
                        }
                        else{
                            defence = i;
                        }
                    }
                    else{
                        if(d_line[graph[i][j]] > 0){
                            defence = i;
                        }
                        else{
                            attack = i;
                            break;
                        }
                    }
                }
            }
            if(attack != -1){
                break;
            }
        }

        if(attack != -1){
            chosen = attack;
        }
        else if(defence != -1){
            chosen = defence;
        }
        else{
            for(int i = 0; i < 9; i++){
                if(d_matrix[priority[i]] == -1){
                    chosen = i;
                    break;
                }
            }
        }

        d_matrix[chosen] = currentPlayer == -1 ? 0 : currentPlayer;

        for(int i = 0; i < 4; i++){
            d_line[graph[chosen][i]] += currentPlayer;
        }
    }
}

// 3 4 5
// 0 1 2 0
// 3 4 5 1 
// 6 7 8 2

__host__ tuple<int *, int *> allocateHostMemory(int size) {
    int *h_matrix = (int *)malloc(sizeof(int) * size * size);
    if (h_matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_FAILURE);
    }
    int *h_line = (int *)malloc(sizeof(int) * size);
    if (h_line == NULL)
    {
        fprintf(stderr, "Failed to allocate host line!\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < size * size; i++){
        h_matrix[i] = -1;
    }
    for(int i = 0; i < 8; i++){
        h_line[i] = 0;
    }
    return {h_matrix, h_line};
}

__host__ tuple<int *, int *> allocateDeviceMemory(int size) {
    int *d_matrix = NULL;
    cudaError_t err = cudaMalloc((void **)&d_matrix, sizeof(int) * size * size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_line = NULL;
    err = cudaMalloc((void **)&d_line, sizeof(int) * 8);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_line (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return {d_matrix, d_line};
}

__host__ void copyFromHostToDevice(int *h_matrix, int *h_line, int *d_matrix, int *d_line, int size) {
    cudaError_t err = cudaMemcpy(d_matrix, h_matrix, sizeof(int) * size * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_line, h_line, sizeof(int) * 8, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy line from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyFromDeviceToHost(int *d_matrix, int *h_matrix, int size)
{
    // Copy the device result int (found index) in device memory to the host result int
    // in host memory.
    cudaError_t err = cudaMemcpy(h_matrix, d_matrix, sizeof(int) * size * size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy int d_matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void deallocateMemory(int *h_matrix, int *d_matrix, int *h_line, int *d_line)
{

    cudaError_t err = cudaFree(d_matrix);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_line);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device int variable d_line (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(h_matrix);


    free(h_line);


}

__host__ void cleanUpDevice()
{
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(int *d_matrix, int *d_line, int numElements, int threadsPerBlock)
{
    // Launch the search CUDA Kernel
    int blocksPerGrid =(1 + threadsPerBlock - 1) / threadsPerBlock;
    

    
    //int result = -1;
    for(int i = 0; i < numElements; i++){
        if(i % 2 == 0){
            tic_tac_toe<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_line, numElements, 1);
        }
        else{
            tic_tac_toe<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_line, numElements, -1);
        }
    }


    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch tic tac toe kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void displayBoard(int *h_matrix, int size) {
    // Host-side function to display the board
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (h_matrix[i * size + j] == -1) {
                cout << " ";
            } else if (h_matrix[i * size + j] == 1) {
                cout << "O";
            } else {
                cout << "X";
            }
            if (j != size - 1) {
                cout << "|";
            }
        }
        cout << endl;
        if (i != size - 1) {
            cout << "-----" << endl;
        }
    }
}

__host__ void showAllCudaDevice(){
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    printf("Number of GPU Devices: %d\n", nDevices);

    // You will need to track the minimum or maximum for one or more device properties, so initialize them here
    int currentChosenDeviceNumber = 0; // Will not choose a device by default 
  
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device Compute Major: %d Minor: %d\n", prop.major, prop.minor);
        printf("  Max Thread Dimensions: [%d][%d][%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Device Clock Rate (KHz): %d\n", prop.clockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Registers Per Block: %d\n", prop.regsPerBlock);
        printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("  Shared Memory Per Block: %zu\n", prop.sharedMemPerBlock);
        printf("  Shared Memory Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
        printf("  Total Constant Memory (bytes): %zu\n", prop.totalConstMem);
        printf("  Total Global Memory (bytes): %zu\n", prop.totalGlobalMem);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    // You can set the current chosen device property based on tracked min/max values
    printf("The chosen GPU device has an index of: %d\n",currentChosenDeviceNumber); 
}

int main(int argc, char *argv[])
{
    showAllCudaDevice();
    int size = 3;
    auto [h_matrix, h_line] = allocateHostMemory(size);
    auto [d_matrix, d_line] = allocateDeviceMemory(size);
    copyFromHostToDevice(h_matrix, h_line, d_matrix, d_line, size);
    int numElements = size * size;
    int threadsPerBlock = 1;
    executeKernel(d_matrix, d_line, numElements, threadsPerBlock);
    copyFromDeviceToHost(d_matrix, h_matrix, size);
    displayBoard(h_matrix, size);
    deallocateMemory(h_matrix, d_matrix, h_line, d_line);
    cleanUpDevice();
    return 0;
}