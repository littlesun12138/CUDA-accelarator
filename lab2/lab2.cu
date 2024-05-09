// LAB 2 SP24

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  // Calculate the row index of the d_P element and d_M
  int Row = blockIdx.y*blockDim.y+threadIdx.y;
  // Calculate the column index of d_P and d_N
  int Col = blockIdx.x*blockDim.x+threadIdx.x;
  if ((Row < numARows) && (Col < numBColumns)) {
      float Pvalue = 0;
  // each thread computes one element of the block sub-matrix
    for (int k = 0; k <numAColumns; ++k){
      Pvalue += A[Row*numAColumns+k] * B[k*numBColumns+Col];
    }
      

  // accumulated dot product is stored in d_P[Row][Col]
    C[Row*numCColumns+Col] = Pvalue;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *dA;
  float *dB;
  float *dC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows*numCColumns * sizeof(float));

  //@@ Allocate GPU memory here
  cudaMalloc((void **) &dA, numARows*numAColumns* sizeof(float));
  cudaMalloc((void **) &dB, numBRows*numBColumns* sizeof(float));
  cudaMalloc((void **) &dC, numCRows*numCColumns* sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(dA, hostA, numARows*numAColumns* sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hostB, numBRows*numBColumns* sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here

  int TILE_WIDTH = 4;
  dim3 dimGrid(ceil((1.0*numCColumns)/TILE_WIDTH),ceil((1.0*numCRows)/TILE_WIDTH), 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);


  //@@ Launch the GPU Kernel here
  // Launch the device computation threads!
  matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, dC, numCRows*numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);
  return 0;
}

