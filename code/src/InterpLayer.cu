#include "InterpLayerPlugin.h"

namespace nvinfer1
{
  InterpLayerPlugin::InterpLayerPlugin(const int cudaThread /*= 512*/)
  : mThreadCount(cudaThread)
  {
  }
  
  InterpLayerPlugin::~InterpLayerPlugin()
  {
  
  }
  
  // create the plugin at runtime from a byte stream
  InterpLayerPlugin::InterpLayerPlugin(const void* data, size_t length)
  {
      using namespace Tn;
      const char *d = reinterpret_cast<const char *>(data), *a = d;
      read(d, mCHW_bottom_0);
      read(d, mCHW_bottom_1);
      read(d, mDataType);
      read(d, mOutputWidth);
      read(d, mOutputHeight);
      read(d, mThreadCount);
  
      //std::cout << "read:" << a << " " << mOutputWidth<< " " <<mOutputHeight<<std::endl;
      assert(d == a + length);
  }
  
  void InterpLayerPlugin::serialize(void* buffer)
  {
      using namespace Tn;
      char* d = static_cast<char*>(buffer), *a = d;
      write(d, mCHW_bottom_0);
      write(d, mCHW_bottom_1);
      write(d, mDataType);
      write(d, mOutputWidth);
      write(d, mOutputHeight);
      write(d, mThreadCount);
  
      //std::cout << "write:" << a << " " << mOutputHeight<< " " <<mOutputWidth<<std::endl;
      assert(d == a + getSerializationSize());
  }
  
  int InterpLayerPlugin::initialize()
  {
      int inputHeight = mCHW_bottom_0.d[1];
      int inputWidth = mCHW_bottom_0.d[2];
      
      mOutputHeight = mCHW_bottom_1.d[1];;
      mOutputWidth = mCHW_bottom_1.d[2];
  
      return 0;
  }
  
  void InterpLayerPlugin::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
  {
      //std::cout << "type " << int(type) << "format " << (int)format <<std::endl;
      assert((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8) && format == PluginFormat::kNCHW);
      mDataType = type;
      
      //std::cout << "configureWithFormat:" <<inputDims[0].d[0]<< " " <<inputDims[0].d[1] << " "<<inputDims[0].d[2] <<std::endl;
      //std::cout << "configureWithFormat:" <<inputDims[1].d[0]<< " " <<inputDims[1].d[1] << " "<<inputDims[0].d[2] <<std::endl;
  }
  
  //it is called prior to any call to initialize().
  Dims InterpLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
  {
      //std::cout <<"Input[0]:" << inputs[0].d[0] << " "<<inputs[0].d[1]<< " "<<inputs[0].d[2]<<std::endl;
      //std::cout <<"Input[1]:" << inputs[1].d[0] << " "<<inputs[1].d[1]<< " "<<inputs[1].d[2]<<std::endl;
      //std::cout <<"nbInputDims : "<<nbInputDims<< " input:" << inputs[0].nbDims << std::endl;
      mCHW_bottom_0 = inputs[0];
      mCHW_bottom_1 = inputs[1];

      mOutputHeight = inputs[1].d[1];
      mOutputWidth = inputs[1].d[2];
      //std::cout << "ouputDims:" << mCHW_bottom_0.d[0] << " " << mOutputHeight << " " << mOutputWidth << std::endl;
      return Dims3(mCHW_bottom_0.d[0], mOutputHeight, mOutputWidth);
  }


 //gpu  
 
  size_t type2size_2(DataType dataType) { 
    size_t _size = 0;
    switch (dataType)
    {
        case DataType::kFLOAT: _size = sizeof(float);break;
        case DataType::kHALF: _size = sizeof(__half);break;
        case DataType::kINT8: _size = sizeof(u_int8_t);break;
        default:std::cerr << "error data type" << std::endl;
    }
    return _size;
  }
  

  // ################# gpu caffe ######################
  /*
  __device__ int translate_idx_2(int ii, int d1, int d2, int d3, int scale_factor) {
    int x, y, z, w;
    w = ii % d3;
    ii = ii/d3;
    z = ii % d2;
    ii = ii/d2;
    y = ii % d1;
    ii = ii/d1;
    x = ii;
    w = w/scale_factor;
    z = z/scale_factor;
    d2 /= scale_factor;
    d3 /= scale_factor;
    return (((x*d1+y)*d2)+z)*d3+w;
  }

  template <typename Dtype>
  __global__ void upscale_2(const Dtype *input, Dtype *output,
          int no_elements, int scale_factor, int d1, int d2, int d3) {
    int ii = threadIdx.x + blockDim.x * blockIdx.x;
    if (ii >= no_elements) return;
    int ipidx = translate_idx_2(ii, d1, d2, d3, scale_factor);
    output[ii]=input[ipidx];
  }

  template <typename Dtype>
  void InterpLayerPlugin::forwardGpu(const Dtype* input, Dtype * output, int IH, int IW, int N,int C,int H ,int W) {
    float mScale = H / IH;
    int numElem = N*C*H*W;
    upscale_2<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount>>>(input,output, numElem, mScale, C, H, W);
  }
  */

 

  template <typename Dtype>
	__global__ void caffe_gpu_interp2_kernel(const int n, const float rheight, const float rwidth,
		const int channels,
		const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
		Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < n) {
			const int w2 = index % width2; // 0:width2-1
			const int h2 = index / width2; // 0:height2-1
			// special case: just copy
			if (height1 == height2 && width1 == width2) {
				const int h1 = h2;
				const int w1 = w2;

        const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += Width1 * Height1;
          pos2 += Width2 * Height2;
        }
				
				return;
			}
			//
			const float h1r = rheight * h2;
			const int h1 = h1r;
			const int h1p = (h1 < height1 - 1) ? 1 : 0;
			const Dtype h1lambda = h1r - h1;
			const Dtype h0lambda = Dtype(1.) - h1lambda;
			//
			const float w1r = rwidth * w2;
			const int w1 = w1r;
			const int w1p = (w1 < width1 - 1) ? 1 : 0;
			const Dtype w1lambda = w1r - w1;
			const Dtype w0lambda = Dtype(1.) - w1lambda;
			//
			
      const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
      Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
      for (int c = 0; c < channels; ++c) {
        pos2[0] =
          h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
          h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
        //printf(pos2[0]);
        pos1 += Width1 * Height1;
        pos2 += Width2 * Height2;
      }
			
		}
	}

  template <typename Dtype>
  void InterpLayerPlugin::forwardGpu(const Dtype* input, Dtype * output, int IH, int IW, int N,int C,int H ,int W){
      const float rheight = (H > 1) ? static_cast<float>(IH - 1) / (H - 1) : 0.f;
      const float rwidth = (W > 1) ? static_cast<float>( IW - 1) / (W - 1) : 0.f;
      const int num_kernels = H * W;
      caffe_gpu_interp2_kernel<Dtype> <<<(num_kernels + mThreadCount - 1) / mThreadCount, mThreadCount >>>
      (num_kernels, rheight,rwidth,N*C,
      input, 0, 0, IH ,IW, IH, IW,
      output, 0, 0, H, W, H, W
      );

  };
  


  int InterpLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
  {
      const int channels = mCHW_bottom_0.d[0];
      const int64_t in_height = mCHW_bottom_0.d[1];
      const int64_t in_width = mCHW_bottom_0.d[2];
      const int64_t out_height = mOutputHeight;
      const int64_t out_width = mOutputWidth;
      int totalElems = batchSize * in_height * in_width * channels;
      
      // Handle no-op resizes efficiently.
      if (out_height == in_height && out_width == in_width) {
          CUDA_CHECK(cudaMemcpyAsync(outputs[0], inputs[0], totalElems * type2size_2(mDataType), cudaMemcpyDeviceToDevice, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          return 0;
      }
      //CUDA_CHECK(cudaStreamSynchronize(stream));

      //printf("hello\n");
      //printf(*inputs[0]->data);
      
       switch (mDataType)
       {
           case DataType::kFLOAT :
              forwardGpu<float>((const float *)inputs[0],(float *)outputs[0], in_height, in_width,batchSize,mCHW_bottom_0.d[0],mOutputHeight,mOutputWidth);
               break;
           //case DataType::kHALF:
           //    forwardGpu<__half>((const __half *)inputs[0],(__half *)outputs[0],in_height, in_width,batchSize,mCHW_bottom_0.d[0],mOutputHeight,mOutputWidth);
           //    break;
           case DataType::kINT8:
               forwardGpu<u_int8_t>((const u_int8_t *)inputs[0],(u_int8_t *)outputs[0],in_height, in_width, batchSize,mCHW_bottom_0.d[0],mOutputHeight,mOutputWidth);
              break;
           default:
               std::cerr << "error data type" << std::endl;
       }
      return 0;    
  };
}
