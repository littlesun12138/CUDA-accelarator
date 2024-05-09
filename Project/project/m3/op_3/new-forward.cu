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
    // int c=Channel;
    // int sharememwidth=TILE_WIDTH+K-1;
    // extern __shared__ float sharedmem[];

    __shared__ float maskmem[TILE_WIDTH*TILE_WIDTH];
    __shared__ float sharedmem[TILE_WIDTH*TILE_WIDTH];
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    int m = blockIdx.x*TILE_WIDTH+threadIdx.x;
    int b = blockIdx.z;
    int W_grid= (Width_out - 1)/ TILE_WIDTH  + 1;
    int h = (blockIdx.y* TILE_WIDTH + threadIdx.y)/ Width_out;
    int w = (blockIdx.y* TILE_WIDTH + threadIdx.y) % Width_out ;
    

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    //SHARED FILLED
    
    
    float sum=0.0;
    
    for (int i = 0; i < ceil((Channel * K * K)/(1.0*TILE_WIDTH)); i++){
        int seq=i * TILE_WIDTH;
        if((threadIdx.y+seq) < Channel * K*K  && m<Map_out){
            maskmem[threadIdx.x*TILE_WIDTH+threadIdx.y]=mask[m* (Channel * K * K) + seq + threadIdx.y];
        }
        else{
                maskmem[threadIdx.x*TILE_WIDTH+threadIdx.y]=0.0;
        }
        //boundary check for  output dim and mapout dim 
        if( ((seq + threadIdx.x) < Channel * K * K) && (blockIdx.y*TILE_WIDTH + threadIdx.y)<Height_out*Width_out){
            
            sharedmem[threadIdx.x*TILE_WIDTH+threadIdx.y]=input[b*(Height * Width*Channel)+ (seq+threadIdx.x)/(K*K)*(Height * Width) + (seq+threadIdx.x)%(K*K)/K*Width+(blockIdx.y* TILE_WIDTH + threadIdx.y)/Width_out*Width + (seq+threadIdx.x)%(K*K)%K+(blockIdx.y* TILE_WIDTH + threadIdx.y)%Width_out];
            
            
        }
        else{
            sharedmem[threadIdx.x*TILE_WIDTH+threadIdx.y]=0.0;
        }
        //this is used for both shared mem sync of mask and input
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += maskmem[threadIdx.x*TILE_WIDTH+k] * sharedmem[k*TILE_WIDTH+threadIdx.y];
        }

        __syncthreads();
    }        
    if( (blockIdx.y*TILE_WIDTH + threadIdx.y)<Height_out*Width_out && m<Map_out){
        output[b* (Map_out * Height_out * Width_out) + m * (Height_out * Width_out) + (h*Width_out + w)]=sum;
    }




    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;           


    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);


    // cudaMemcpyToSymbol(MASK, host_mask, Map_out * Channel * K * K * sizeof(float));
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
    // float *unroll_input_ptr;  
    // cudaMalloc((void **) &unroll_input_ptr, Batch * Channel *K*K* Height_out * Width_out * sizeof(float));    
    // dim3 dimBlock(TILE_WIDTH, 1, 1);
    // dim3 dimGrid( ceil( Height_out* Width_out*Channel / (1.0*TILE_WIDTH)),1, Batch);
    // unroll<<<dimGrid,dimBlock>>>(unroll_input_ptr,device_input,Channel, Height,Width, K);
    // cudaDeviceSynchronize();

    // //shared mem can reuse input feature in different maps,so need to use mapout as one dimension
    // dim3 dimBlock1(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 dimGrid1( ceil(Width_out * Height_out /(1.0*TILE_WIDTH) ),ceil(Map_out/(1.0*TILE_WIDTH)) , Batch);

    // //shared mem need to be larger.
    // // size_t smsize=sizeof(float) *TILE_WIDTH*TILE_WIDTH;


    // // conv_forward_kernel<<<dimGrid1,dimBlock1,smsize>>>(device_output,unroll_input_ptr,device_mask,Batch,Map_out,Channel,Height,Width,K);
    // conv_forward_kernel<<<dimGrid1,dimBlock1>>>(device_output,unroll_input_ptr,device_mask,Batch,Map_out,Channel,Height,Width,K);
    // cudaFree(unroll_input_ptr);
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    // cudaMalloc((void **) &unroll_input_ptr, Batch * Channel *K*K* Height_out * Width_out * sizeof(float));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil(Map_out/(1.0*TILE_WIDTH)), ceil(Width_out * Height_out /(1.0*TILE_WIDTH) ), Batch);

    // size_t shared_mem_size = TILE_WIDTH  * TILE_WIDTH  * sizeof(float);

    // Launch the fused kernel
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

    // // Check for errors
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height-K+1) * (Width-K+1) * sizeof(float), cudaMemcpyDeviceToHost);
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