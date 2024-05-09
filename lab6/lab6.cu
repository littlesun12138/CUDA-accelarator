// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void add(float *input, float *output,int len, float *val){
  __shared__ float single;

  int startindex=2*blockIdx.x*blockDim.x+ threadIdx.x;
  //first thread
  if(threadIdx.x==0){
    if(blockIdx.x!=0){
      single=val[blockIdx.x-1];
    }
    else{
      single=0.0;
    }
  }
  __syncthreads();
  if (startindex<len){
    output[startindex] = input[startindex] +single;
  }
  if ((startindex + blockDim.x)<len){
    output[startindex+blockDim.x]=single+input[startindex+blockDim.x];
  }
  // __syncthreads();
}
__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];

  int start=2*blockIdx.x*blockDim.x + threadIdx.x;
  int firststride = blockDim.x;
  int secondstride =1;
  if(start < len){
    T[threadIdx.x]=input[start];
  }
  else{
    T[threadIdx.x]=0.0;
  }
  if(start+firststride < len){
    T[blockDim.x+threadIdx.x]=input[start+firststride];
  }
  else{
    T[blockDim.x+threadIdx.x]=0.0;
  }

  while(secondstride<2*BLOCK_SIZE){
    __syncthreads();
    int id = (threadIdx.x + 1)*secondstride*2 - 1;
    if (id<2*BLOCK_SIZE && id>=secondstride){
        T[id]+=T[id-secondstride];
    }
    secondstride = secondstride*2;
    
  }


  secondstride = BLOCK_SIZE/2;
  while (secondstride > 0)
  {
    __syncthreads();
    int index=(threadIdx.x+1)*secondstride*2 - 1;
    if ((index+secondstride)<2*BLOCK_SIZE)
      T[index+secondstride]+=T[index];
    secondstride=secondstride/2; 
    
  } 
  __syncthreads();

  int ind = 2*blockIdx.x*blockDim.x+ threadIdx.x;
  if (ind<len){
    output[ind] = T[threadIdx.x];
  }
  if ((ind+blockDim.x)<len){
    output[ind+blockDim.x]=T[blockDim.x+threadIdx.x];
  }

}

__global__ void sscan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];

  int start=2*blockDim.x*(threadIdx.x+1)-1;
  int firststride =blockDim.x*2;
  int secondstride =1;
  if(start < len){
    T[threadIdx.x]=input[start];
  }
  else{
    T[threadIdx.x]=0.0;
  }
  if(start+firststride < len){
    T[blockDim.x+threadIdx.x]=input[start+firststride];
  }
  else{
    T[blockDim.x+threadIdx.x]=0.0;
  }
  

  while(secondstride<2*BLOCK_SIZE){
    __syncthreads();
    int id = (threadIdx.x + 1)*secondstride*2 - 1;
    if (id<2*BLOCK_SIZE && id>=secondstride){
        T[id]+=T[id-secondstride];
    }
    secondstride = secondstride*2;
    
  }


  secondstride = BLOCK_SIZE/2;
  while (secondstride > 0)
  {
    __syncthreads();
    int index=(threadIdx.x+1)*secondstride*2 - 1;
    if ((index+secondstride)<2*BLOCK_SIZE)
      T[index+secondstride]+=T[index];
    secondstride=secondstride/2; 
    
  } 
  __syncthreads();

  int ind = 2*blockIdx.x*blockDim.x+ threadIdx.x;
  if (ind<len){
    output[ind] = T[threadIdx.x];
  }
  if ((ind+blockDim.x)<len){
    output[ind+blockDim.x]=T[blockDim.x+threadIdx.x];
  }

}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *devicebuf;
  float *devicesum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicebuf, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicesum, 2*numElements * sizeof(float)));
  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(BLOCK_SIZE*2.0)),1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, devicebuf, numElements);
  cudaDeviceSynchronize();
  sscan<<<1, dimBlock>>>(devicebuf, devicesum, numElements);
  cudaDeviceSynchronize();
  add<<<dimGrid, dimBlock>>>(devicebuf, deviceOutput, numElements,devicesum );
  cudaDeviceSynchronize();
  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(devicebuf);
  cudaFree(devicesum);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

