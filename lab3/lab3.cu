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

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  // Loop over the M and N tiles required to compute the P element
  // The code assumes that the Width is a multiple of TILE_WIDTH!
  for (int q = 0; q < (numAColumns-1)/TILE_WIDTH+1; ++q) {
      // Collaborative loading of M and N tiles into shared memory
      //A check range
      if(Row<numARows && q*TILE_WIDTH+tx<numAColumns){
        subTileM[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH+tx];
      }
      else{
        subTileM[ty][tx]=0.0;
      }
      //B check range
      if( Col < numBColumns && q*TILE_WIDTH+ty<numBRows){
        subTileN[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns+Col];
      }
      else{
        subTileN[ty][tx]=0.0;
      }
      __syncthreads();
      for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += subTileM[ty][k] * subTileN[k][tx];
      __syncthreads();
  }

  if(Row<numCRows && Col<numCColumns){
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
  dim3 dimGrid((numCColumns-1)/TILE_WIDTH+1,(numCRows-1)/TILE_WIDTH+1, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);  

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(dA, dB, dC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
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

  //@@ Free the hostC matrix
  free(hostC);
  return 0;
}
