#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 4
//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
int tx = threadIdx.x;
int ty = threadIdx.y;
int tz = threadIdx.z;
int row_o = blockIdx.y * TILE_WIDTH + ty;
int col_o = blockIdx.x * TILE_WIDTH + tx;
int dep_o = blockIdx.z * TILE_WIDTH + tz;

//radius: maxwidth-1/2
int row_i = row_o - 1;
int col_i = col_o - 1;
int dep_i = dep_o - 1;

__shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

 if (0 <= row_i && y_size > row_i && 0 <= col_i && x_size > col_i && 0 <= dep_i && z_size > dep_i) { 
    N_ds[tz][ty][tx] = input[dep_i*x_size*y_size+row_i*x_size+col_i];
 }
 else{
    N_ds[tz][ty][tx] = 0.0f;
 }
  __syncthreads();

 float Pvalue = 0.0f;
  if(tx<TILE_WIDTH &&ty<TILE_WIDTH && tz<TILE_WIDTH ){
    for (int k = 0; MASK_WIDTH > k; k++) {
      for (int j = 0; MASK_WIDTH > j; j++) {
        for (int i = 0; MASK_WIDTH > i; i++) {
            Pvalue+=N_ds[k+tz][j+ty][i+tx]*Mc[k][j][i];
        }
      }
    }

  //boundary check
    if(row_o<y_size && col_o<x_size && dep_o<z_size){
      output[dep_o*x_size*y_size+row_o*x_size+col_o]=Pvalue;
    }
  }  
  __syncthreads();
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *dInput;
  float *dOutput;
  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &dInput,(inputLength-3) * sizeof(float));
  cudaMalloc((void **) &dOutput,(inputLength-3) * sizeof(float));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(dInput, &hostInput[3], z_size*y_size*x_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH * sizeof(float));


  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0*x_size) / TILE_WIDTH), ceil((1.0*y_size) / TILE_WIDTH), ceil((1.0*z_size) / TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(dInput, dOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], dOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(dInput);
  cudaFree(dOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

