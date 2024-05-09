// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void tochar_kernel(float *input,unsigned char *output, int w, int h){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel=blockIdx.z;
    if(row<h &&col<w){
      //thread inside photo range
      int i=channel*w*h +row*w+col;
      output[i]=(unsigned char)255*input[i];
    }
  
}
__global__ void togray_kernel(unsigned char *input,unsigned char *output, int w, int h){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int channel=blockIdx.z;
    if(row<h &&col<w){
      //thread inside photo range
      // int i=channel*w*h +row*w+col;
      int idx = row*w+col;
		
      unsigned char r = input[3*idx];
      unsigned char g = input[3*idx + 1];
      unsigned char b = input[3*idx + 2];
      output[idx]=(unsigned char)(0.21*r + 0.71*g + 0.07*b);
    }
  
}


__global__ void histo_kernel(unsigned char *input,unsigned int *output, int w, int h){
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  int i = threadIdx.x + threadIdx.y * blockDim.x;
  

  if(i < HISTOGRAM_LENGTH) {
    histo_private[i]=0;
  }
  __syncthreads();


  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // int channel=blockIdx.z;
  if(row<h &&col<w){
    //thread inside photo range
    // int i=channel*w*h +row*w+col;
    int idx = row*w+col;
    atomicAdd(&(histo_private[input[idx]]), 1);
    
  }
  __syncthreads();
    if (i < HISTOGRAM_LENGTH) {
        atomicAdd(&(output[i]), histo_private[i]);
    }
}

__global__ void cdf_kernel(unsigned int *input,float *output, int w, int h){
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  //here we use 1d block
  int i = threadIdx.x;
  if (i < HISTOGRAM_LENGTH) {
    cdf[i] = input[i];
  }
  __syncthreads();


  for (int stride = 1; stride<=HISTOGRAM_LENGTH/2;stride*=2) {
    int idx =(i+1)*stride*2-1;
    if (idx < HISTOGRAM_LENGTH && (idx-stride)<HISTOGRAM_LENGTH) {
      cdf[idx] += cdf[idx-stride];
    }
    __syncthreads();
  }


  for (int stride = HISTOGRAM_LENGTH/4; stride>0; stride/=2) {
    int idx =(i+1)*stride*2-1;
    if ((idx + stride) < HISTOGRAM_LENGTH) {
      cdf[idx+stride] += cdf[idx];
    }
    __syncthreads();
  }
  if (i < HISTOGRAM_LENGTH) {
    output[i] = cdf[i]/(1.0f*w*h);
  }

}

__global__ void equal_kernel(unsigned char* input, float *cdf, int w, int h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel=blockIdx.z;

    if (row < h && col < w) {
        int i=channel*w*h +row*w+col;
      
        float accumulate = 255 *(cdf[input[i]]-cdf[0])/(1.0-cdf[0]);
        float mini = min(max(accumulate, 0.0), 255.0);

        input[i] = (unsigned char)mini;
    }
}

__global__ void float_kernel(unsigned char *input, float *output, int w, int h){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel=blockIdx.z;
    int i=channel*w*h +row*w+col;
    if (row < h && col < w){
      output[i] = (float)(input[i]/255.0);
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *dinput;
  float *doutput;
  unsigned char *dcolor;
  unsigned char *dgray;
  unsigned int *dhisto;
  float *dcdf;




  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ insert code here
  cudaMalloc((void**) &dinput,sizeof(float) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**) &doutput, sizeof(float) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**) &dcolor, sizeof(unsigned char) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**) &dgray, sizeof(unsigned char) * imageWidth * imageHeight);
  cudaMalloc((void**) &dhisto, sizeof(unsigned int) * HISTOGRAM_LENGTH);
  cudaMalloc((void**) &dcdf, sizeof(float) * HISTOGRAM_LENGTH);
  cudaMemset((void*) dhisto, 0 , HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void*) dcdf, 0 , HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(dinput, hostInputImageData,sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyHostToDevice);


  dim3 dimGrid(ceil( imageWidth/32.0), ceil( imageHeight/32.0), imageChannels);
  dim3 dimBlock(32, 32, 1);
  tochar_kernel<<<dimGrid, dimBlock>>>(dinput,dcolor, imageWidth , imageHeight);
  cudaDeviceSynchronize();

  dimGrid = dim3(ceil( imageWidth/32.0), ceil( imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  togray_kernel<<<dimGrid,dimBlock>>>(dcolor , dgray, imageWidth , imageHeight);
  cudaDeviceSynchronize();

  histo_kernel<<<dimGrid,dimBlock>>>(dgray, dhisto, imageWidth , imageHeight);
  cudaDeviceSynchronize();

  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(256, 1, 1);
  cdf_kernel<<<dimGrid,dimBlock>>>(dhisto, dcdf,imageWidth , imageHeight);
  cudaDeviceSynchronize();

  dimGrid = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  equal_kernel<<<dimGrid, dimBlock>>>(dcolor, dcdf, imageWidth , imageHeight);
  cudaDeviceSynchronize();
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  float_kernel<<<dimGrid, dimBlock>>>(dcolor ,doutput, imageWidth , imageHeight);
  cudaDeviceSynchronize();
  
  cudaMemcpy(hostOutputImageData, doutput, sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyDeviceToHost);
  //@@ insert code here
  
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);
  cudaFree(dinput);
  cudaFree(doutput);
  cudaFree(dcolor);
  cudaFree(dgray);
  cudaFree(dhisto);
  cudaFree(dcdf);
  free(hostOutputImageData);
  free(hostInputImageData);


  // wbImage_setData(outputImage, hostOutputImageData);
  return 0;
}

