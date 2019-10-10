#include "PadChannelLayer.h"

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

namespace nvinfer1
{
  PadChannelPlugin::PadChannelPlugin(const int cudaThread /*= 512*/):mThreadCount(cudaThread)
  {
  }
  
  PadChannelPlugin::~PadChannelPlugin()
  {
  
  }
  
  // create the plugin at runtime from a byte stream
  PadChannelPlugin::PadChannelPlugin(const void* data, size_t length)
  {
      using namespace Tn;
      const char *d = reinterpret_cast<const char *>(data), *a = d;
      read(d, mCHW_bottom);
      read(d, mDataType);
      read(d, mThreadCount);

      assert(d == a + length);
  }
  
  void PadChannelPlugin::serialize(void* buffer)
  {
      using namespace Tn;
      char* d = static_cast<char*>(buffer), *a = d;
      write(d, mCHW_bottom);
      write(d, mDataType);
      write(d, mThreadCount);

      assert(d == a + getSerializationSize());
  }
  
  int PadChannelPlugin::initialize()
  {
    mThreadCount = 512;
      return 0;
  }
  
  void PadChannelPlugin::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
  {
      //std::cout << "type " << int(type) << "format " << (int)format <<std::endl;
      assert((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8) && format == PluginFormat::kNCHW);
      mDataType = type;
      
      //std::cout << "configureWithFormat:" <<inputDims[0].d[0]<< " " <<inputDims[0].d[1] << " "<<inputDims[0].d[2] <<std::endl;
  }
  
  //it is called prior to any call to initialize().
  Dims PadChannelPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
  {
      mCHW_bottom = inputs[0];

      return Dims3(mCHW_bottom.d[0] * 2, inputs[0].d[1], inputs[0].d[2]);
  }


 //gpu  
 

  // ################# gpu caffe ######################
template <typename Dtype>
	__global__ void pad_forward_kernel(const int dst_count, const int src_channels, const int dst_channels,
		const int dim, const Dtype* src, Dtype* dst){
		CUDA_KERNEL_LOOP(index, dst_count)
		{
			int num = index / (dim * dst_channels);
			int dst_c = index / dim % dst_channels;
			int pixel_pos = index % dim;
			if (dst_c < src_channels)
				dst[index] = src[num * src_channels * dim + dst_c * dim + pixel_pos];
			else
				dst[index] = Dtype(0);
		}
	}


  template <typename Dtype>
  void PadChannelPlugin::forwardGpu(const Dtype* input, Dtype * output, int IH, int IW, int N, int C){
      const int num_kernels = N * (C * 2) * IH * IW;
      pad_forward_kernel<<<(num_kernels + mThreadCount - 1) / mThreadCount, mThreadCount >>>
      (num_kernels, C, C*2, IH*IW, input, output);

  };
  


  int PadChannelPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
  {
      const int channels = mCHW_bottom.d[0];
      const int64_t in_height = mCHW_bottom.d[1];
      const int64_t in_width = mCHW_bottom.d[2];

      //printf("hello\n");
      //printf(*inputs[0]->data);
      
       switch (mDataType)
       {
           case DataType::kFLOAT :
              forwardGpu<float>((const float *)inputs[0],(float *)outputs[0], in_height, in_width,batchSize, mCHW_bottom.d[0]);
               break;
           case DataType::kHALF:
               forwardGpu<__half>((const __half *)inputs[0],(__half *)outputs[0],in_height, in_width,batchSize, mCHW_bottom.d[0]);
               break;
           case DataType::kINT8:
               forwardGpu<u_int8_t>((const u_int8_t *)inputs[0],(u_int8_t *)outputs[0],in_height, in_width, batchSize, mCHW_bottom.d[0]);
              break;
           default:
               std::cerr << "error data type" << std::endl;
       }
      return 0;    
  };
}
