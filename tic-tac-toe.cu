#include "tic-tac-toe.h"

__global__ void tic_tac_toe(int *d_matrix, int *d_line, int numElements, int currentPlayer)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < 1)
    {
        int priority[9] = {4, 0, 2, 6, 8, 1, 3, 5, 7};
        int graph[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};

        int attack = -1;
        int defence = -1;
        int chosen = -1;

        for(int i = 0; i < 8; i++){
            if(d_line[i] == 2){
                if(currentPlayer == 1){
                    attack = i;
                }
                else{
                    defence = i;
                }
            }
            else if(d_line[i] == -2){
                if(currentPlayer == -1){
                    attack = i;
                }
                else{
                    defence = i;
                }
            }
        }

        if(attack != -1){
            for(int i = 0; i < 3; i++){
                if(d_matrix[graph[attack][i]] == -1){
                    chosen = graph[attack][i];
                    d_matrix[graph[attack][i]] = currentPlayer == -1 ? 0 : 1;
                    break;
                }
            }
        }
        else if(defence != -1){
            for(int i = 0; i < 3; i++){
                if(d_matrix[graph[defence][i]] == -1){
                    chosen = graph[defence][i];
                    d_matrix[graph[defence][i]] = currentPlayer == -1 ? 0 : 1;
                    break;
                }
            }
        }
        else{
            for(int i = 0; i < 9; i++){
                if(d_matrix[priority[i]] == -1){
                    chosen = priority[i];
                    d_matrix[priority[i]] = currentPlayer == -1 ? 0 : 1;
                    break;
                }
            }
        }

        switch(chosen){
            case 0:
                d_line[0] += currentPlayer;
                d_line[3] += currentPlayer;
                d_line[6] += currentPlayer;
                break;
            case 1:
                d_line[0] += currentPlayer;
                d_line[4] += currentPlayer;
                break;
            case 2:
                d_line[0] += currentPlayer;
                d_line[5] += currentPlayer;
                d_line[7] += currentPlayer;
                break;
            case 3:
                d_line[1] += currentPlayer;
                d_line[3] += currentPlayer;
                break;
            case 4:
                d_line[1] += currentPlayer;
                d_line[4] += currentPlayer;
                d_line[6] += currentPlayer;
                d_line[7] += currentPlayer;
                break;
            case 5:
                d_line[1] += currentPlayer;
                d_line[5] += currentPlayer;
                break;
            case 6:
                d_line[2] += currentPlayer;
                d_line[3] += currentPlayer;
                d_line[7] += currentPlayer;
                break;
            case 7:
                d_line[2] += currentPlayer;
                d_line[4] += currentPlayer;
                break;
            case 8:
                d_line[2] += currentPlayer;
                d_line[5] += currentPlayer;
                d_line[6] += currentPlayer;
                break;
        }
    }
}

// 3 4 5
// 0 1 2 0
// 3 4 5 1 
// 6 7 8 2

__global__ void test(int *d_matrix, int *d_line, int numElements, int currentPlayer) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx < 1){
        for(int i = 0; i < numElements; i++){
            if(d_matrix[i] == -1){
                d_matrix[i] = currentPlayer == -1 ? 0 : 1;
                break;
            }
        }
    }
}



__host__ tuple<int *, int *> allocateHostMemory(int matrix_size) {
    size_t size = matrix_size * matrix_size * sizeof(int);
    int *h_matrix = (int *)malloc(size);
    if (h_matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_FAILURE);
    }
    int *h_line = (int *)malloc(sizeof(int) * 8);
    if (h_line == NULL)
    {
        fprintf(stderr, "Failed to allocate host line!\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < matrix_size * matrix_size; i++){
        h_matrix[i] = -1;
    }
    for(int i = 0; i < 8; i++){
        h_line[i] = 0;
    }
    return {h_matrix, h_line};
}

__host__ tuple<int *, int *> allocateDeviceMemory(int matrix_size) {
    size_t size = matrix_size * matrix_size * sizeof(int);
    int *d_matrix = NULL;
    cudaError_t err = cudaMalloc((void **)&d_matrix, size);
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

__host__ void copyFromHostToDevice(int *h_matrix, int *h_line, int *d_matrix, int *d_line, int matrix_size) {
    size_t size = matrix_size * matrix_size * sizeof(int);
    cudaError_t err = cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
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

__host__ void copyFromDeviceToHost(int *d_matrix, int *h_matrix, int *d_line, int *h_line, int matrix_size)
{
    // Copy the device result int (found index) in device memory to the host result int
    // in host memory.
    size_t size = matrix_size * matrix_size * sizeof(int);
    cudaError_t err = cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy int d_matrix from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_line, d_line, sizeof(int) * 8, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy int d_line from device to host (error code %s)!\n", cudaGetErrorString(err));
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

__host__ void executeKernel(int *d_matrix, int *d_line, int numElements, int threadsPerBlock, int *h_matrix, int *h_line)
{

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);

    // Launch the search CUDA Kernel
    int blocksPerGrid =(1 + threadsPerBlock - 1) / threadsPerBlock;
    
    int result = -1;
    for(int i = 0; i < numElements; i++){
        if(i % 2 == 0){
            if(num_gpus > 1){
                tic_tac_toe<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_line, numElements, 1);
                cudaDeviceSynchronize();
                cudaSetDevice(1);    
            }
            else{
                cudaStream_t stream_player1;
                cudaStreamCreate(&stream_player1);
                tic_tac_toe<<<blocksPerGrid, threadsPerBlock, 0, stream_player1>>>(d_matrix, d_line, numElements, 1);
                cudaDeviceSynchronize();
                cudaStreamDestroy(stream_player1);
            }
        }
        else{
            if(num_gpus > 1){
                tic_tac_toe<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_line, numElements, -1);
                cudaDeviceSynchronize();
                cudaSetDevice(0);    
            }
            else{
                cudaStream_t stream_player2;
                cudaStreamCreate(&stream_player2);
                tic_tac_toe<<<blocksPerGrid, threadsPerBlock, 0, stream_player2>>>(d_matrix, d_line, numElements, -1);
                cudaDeviceSynchronize();
                cudaStreamDestroy(stream_player2);
            }
        }
        copyFromDeviceToHost(d_matrix, h_matrix, d_line, h_line, sqrt(numElements));

        for(int l = 0; l < 8; l++){
            if(abs(h_line[l]) == 3){
                result = h_line[l] > 0 ? 1 : 0;
                break;
            }
        }

        if(result != -1){
            break;
        }
    }


    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch tic tac toe kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    switch(result){
        case -1:
            cout << "Result: Tie" << endl;
            break;
        case 0:
            cout << "Result: Player2 Win!" << endl;
            break;
        case 1:
            cout << "Result: Player1 Win!" << endl;
            break;
    }

}

__host__ void displayBoard(int *h_matrix, int matrix_size) {
    // Host-side function to display the board
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            if (h_matrix[i * matrix_size + j] == -1) {
                cout << " ";
            } else if (h_matrix[i * matrix_size + j] == 1) {
                cout << "O";
            } else if (h_matrix[i * matrix_size + j] == 0){
                cout << "X";
            }
            if (j != matrix_size - 1) {
                cout << "|";
            }
        }
        cout << endl;
        if (i != matrix_size - 1) {
            cout << "-----" << endl;
        }
    }
    cout << endl;
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


    int matrix_size = 3;
    auto [h_matrix, h_line] = allocateHostMemory(matrix_size);
    auto [d_matrix, d_line] = allocateDeviceMemory(matrix_size);
    copyFromHostToDevice(h_matrix, h_line, d_matrix, d_line, matrix_size);
    int numElements = matrix_size * matrix_size;
    int threadsPerBlock = 1;
    int input;
    do{
        executeKernel(d_matrix, d_line, numElements, threadsPerBlock, h_matrix, h_line);
        copyFromDeviceToHost(d_matrix, h_matrix, d_line, h_line, matrix_size);
        displayBoard(h_matrix, matrix_size);

        cout << "Pres Any Key to Rematch Or Enter -1 to Exit" << endl;
        cout << "-------------------------------------------" << endl;
        cin >> input;
    }while(input != -1);
    deallocateMemory(h_matrix, d_matrix, h_line, d_line);
    cleanUpDevice();
    return 0;
}