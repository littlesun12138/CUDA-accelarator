#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    int m = blockIdx.x;
    int b = blockIdx.z;
    int W_grid= (Width_out - 1)/ TILE_WIDTH  + 1;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    if(w < Width_out && h < Height_out){
        float acc = 0.0f;
        for(int i = 0; i < Channel; i++){
            for(int j = 0; j < K; j++){                    
                for(int k = 0; k < K; k++){
                    acc += in_4d(b, i , h + j, w + k) * mask_4d(m, i , j , k); 
                }
            }
        }
        out_4d(b, m, h, w) = acc;
    }
    __syncthreads();    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    //create 3 streams.
    cudaStream_t stream0, stream1, stream2;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Allocate memory and copy over the relevant data structures to the GPU
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;   
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    int SegSize= Channel * Height * Width;
    int count=0;

    int outSegSize=Map_out*Height_out*Width_out;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Map_out, ceil(1.0 * Height_out / TILE_WIDTH) * ceil(1.0 * Width_out / TILE_WIDTH), 1);    
    for (int i=0; i<Batch && i+1<Batch && i+2<Batch; i+=3) {

        cudaMemcpyAsync(*device_input_ptr+i*SegSize, host_input+i*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
        
        cudaMemcpyAsync(*device_input_ptr+(i+1)*SegSize, host_input+(i+1)*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice, stream1);
        
        cudaMemcpyAsync(*device_input_ptr+(i+2)*SegSize, host_input+(i+2)*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice, stream2);
        
        
        conv_forward_kernel<<<dimGrid,dimBlock,0,stream0>>>(*device_output_ptr+i*outSegSize,*device_input_ptr+i*SegSize,*device_mask_ptr,Batch,Map_out,Channel,Height,Width,K);
        conv_forward_kernel<<<dimGrid,dimBlock,0,stream1>>>(*device_output_ptr+(i+1)*outSegSize,*device_input_ptr+(i+1)*SegSize,*device_mask_ptr,Batch,Map_out,Channel,Height,Width,K);
        conv_forward_kernel<<<dimGrid,dimBlock,0,stream2>>>(*device_output_ptr+(i+2)*outSegSize,*device_input_ptr+(i+2)*SegSize,*device_mask_ptr,Batch,Map_out,Channel,Height,Width,K);


        cudaMemcpyAsync((void *)(host_output+i*outSegSize), *device_output_ptr+i*outSegSize, outSegSize*sizeof(float),cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync((void *)(host_output+(i+1)*outSegSize), *device_output_ptr+(i+1)*outSegSize,outSegSize*sizeof(float),cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync((void *)(host_output+(i+2)*outSegSize), *device_output_ptr+(i+2)*outSegSize,outSegSize*sizeof(float),cudaMemcpyDeviceToHost, stream2);
        count=i;
    }
    if(count+3==Batch-1){
        cudaMemcpyAsync(*device_input_ptr+(count+3)*SegSize, host_input+(count+3)*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
        conv_forward_kernel<<<dimGrid,dimBlock,0,stream0>>>(*device_output_ptr+(count+3)*outSegSize,*device_input_ptr+(count+3)*SegSize,*device_mask_ptr,Batch,Map_out,Channel,Height,Width,K);
        cudaMemcpyAsync((void *)(host_output+(count+3)*outSegSize), *device_output_ptr+(count+3)*outSegSize, outSegSize*sizeof(float),cudaMemcpyDeviceToHost, stream0);
    }
    else if(count+3==Batch-2){
        cudaMemcpyAsync(*device_input_ptr+(count+3)*SegSize, host_input+(count+3)*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(*device_input_ptr+(count+4)*SegSize, host_input+(count+4)*SegSize,SegSize*sizeof(float),cudaMemcpyHostToDevice, stream1);
        conv_forward_kernel<<<dimGrid,dimBlock,0,stream0>>>(*device_output_ptr+(count+3)*outSegSize,*device_input_ptr+(count+3)*SegSize,*device_mask_ptr,Batch,Map_out,Channel,Height,Width,K);
        conv_forward_kernel<<<dimGrid,dimBlock,0,stream1>>>(*device_output_ptr+(count+4)*outSegSize,*device_input_ptr+(count+4)*SegSize,*device_mask_ptr,Batch,Map_out,Channel,Height,Width,K);
        cudaMemcpyAsync((void *)(host_output+(count+3)*outSegSize), *device_output_ptr+(count+3)*outSegSize, outSegSize*sizeof(float),cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync((void *)(host_output+(count+4)*outSegSize), *device_output_ptr+(count+4)*outSegSize,outSegSize*sizeof(float),cudaMemcpyDeviceToHost, stream1);

    }
    // cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);


    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);


    
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    // int Height_out = Height - K + 1;
    // int Width_out = Width - K + 1;
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 dimGrid(Map_out, ceil(1.0 * Height_out / TILE_WIDTH) * ceil(1.0 * Width_out / TILE_WIDTH), Batch);
    // conv_forward_kernel<<<dimGrid,dimBlock>>>(device_output,device_input,device_mask,Batch,Map_out,Channel,Height,Width,K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    // cudaMemcpy(host_output, device_output, Batch * Map_out * (Height-K+1) * (Width-K+1) * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}