// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<len){
    out[i] = in1[i] + in2[i];
  } 
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *hostInput1_d;
  float *hostInput2_d;
  float *hostOutput_d;
  // float *hostOutput1;
  // float *hostOutput2; 
  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  // hostOutput1 = (float *)malloc(inputLength * sizeof(float));
  // hostOutput2 = (float *)malloc(inputLength * sizeof(float));  
  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here

  cudaMalloc((void **) &hostInput1_d, inputLength * sizeof(float));
  cudaMalloc((void **) &hostInput2_d, inputLength * sizeof(float));
  cudaMalloc((void **) &hostOutput_d, inputLength * sizeof(float));
  //@@ Copy memory to the GPU here
  cudaMemcpy(hostInput1_d, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(hostInput2_d, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/256.0), 1, 1);
  // if (0 != (inputLength % 256)) { DimGrid.x++; }
  dim3 DimBlock(256, 1, 1);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(hostInput1_d, hostInput2_d,hostOutput_d, inputLength);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  // printf("%f",hostOutput_d[0]);
  // printf("%f",hostOutput[0]);  
  cudaMemcpy(hostOutput, hostOutput_d, inputLength * sizeof(float), cudaMemcpyDeviceToHost);  
  // cudaMemcpy(hostOutput1, hostInput1_d, inputLength * sizeof(float), cudaMemcpyDeviceToHost);  
  // cudaMemcpy(hostOutput2, hostInput2_d, inputLength * sizeof(float), cudaMemcpyDeviceToHost);  
  // printf("testing\n");
  // printf("%f\n",hostOutput[0]);
  // printf("%f\n",hostOutput1[0]);
  // printf("%f\n",hostOutput2[0]);
  //@@ Free the GPU memory here
  cudaFree(hostInput1_d);
  cudaFree(hostInput2_d);
  cudaFree(hostOutput_d);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
