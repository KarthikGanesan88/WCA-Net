#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cudnn.h>
#include "fp_mult.h"

namespace py = pybind11;

__global__ void matmul_cuda_int8(
  const float *image,
  const float image_s,
  const int image_zp,
  const float *weight,
  const float weight_s,
  const int weight_zp,
	const float *bias,
  float *output,
  const int m,
  const int n,
  const int k,
  const int batch_size,
  const int appx_mode
){
    int img = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    // int product = 0;
    if( col < k && row < m && img < batch_size){
        //printf("%d\n", img);
        for(int i = 0; i < n; i++){
				    float a = image[(img*m*n)+(row*n) + i] - image_zp;
				    float b = weight[i * k + col] - weight_zp;
						// product = a * b;
            switch (appx_mode){
              case PRECISE: sum += (a*b); break;
              default: sum += int_appx_mul((short int)a, (short int)b, 16, appx_mode); break;
            }
						// if ((product_appx - product) != 0.0f)
						// 	printf("(%f*%f)=%f, !=%f\n", a, b, product, product_appx);
						//sum += FP_appx_mul(a,b);
            //sum += image[(img*m*n)+(row*n) + i] * weight[i * k + col];
						//printf("[%d,%d,%d,%d]:(%1.3f*%1.3f)\n", blockIdx.y, blockIdx.x, row, col, \
																image[(img*m*n)+(row*n) + i], weight[i * k + col]);
        }
        output[(img*m*k)+(row*k) + col] = (sum*image_s*weight_s) + bias[col];
        // output[(img*m*k)+(row*k) + col] = (final>0.0f)?final:0.0f;
    }
}


__global__ void matmul_cuda_float(
  const float *image,
  const float *weight,
	const float *bias,
  float *output,
  const int m,
  const int n,
  const int k,
  const int batch_size,
  const int appx_mode,
  const bool use_bias
){
    int img = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if ((img==0) && (row==0) && (col==0))
    //   printf("Appx mode (CUDA):%d\n", appx_mode);

    float sum = 0.0f;
    if( col < k && row < m && img < batch_size){
        //printf("%d\n", img);
        for(int i = 0; i < n; i++){
				    float a = image[(img*m*n)+(row*n) + i];
				    float b = weight[i * k + col];
						switch (appx_mode){
              case PRECISE: sum += (a*b); break;
                // if ((img==0) && (row==0) && (col==0))
                //   printf("precise mult\n");
              default: sum += fp_appx_mul(a,b, appx_mode); break;
                // if ((img==0) && (row==0) && (col==0))
                //   printf("appx mult\n");
            }
						// if ((product_appx - product) != 0.0f)
						// 	printf("(%f*%f)=%f, !=%f\n", a, b, product, product_appx);
						//sum += FP_appx_mul(a,b);
            //sum += image[(img*m*n)+(row*n) + i] * weight[i * k + col];
						//printf("[%d,%d,%d,%d]:(%1.3f*%1.3f)\n", blockIdx.y, blockIdx.x, row, col, \
																image[(img*m*n)+(row*n) + i], weight[i * k + col]);
        }
        if (use_bias == true) {
            sum += bias[col];
        }

        output[(img*m*k)+(row*k) + col] = sum;
    }
}

torch::Tensor conv_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k,
	int b,
  int appx_mode,
  bool use_bias
) {

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({b, m, k}, options);

	//std::cout << "Input(CUDA):\n" << input << std::endl;
	//std::cout << "weight(CUDA):\n" << weight << std::endl;

	unsigned int block_size = 8; // Use this block size to not exceed 1024 threads
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;
	unsigned int grid_images = (b + block_size - 1) / block_size;
	//printf("Grid Rows: %d, Grid columns: %d, Grid images: %d\n", grid_rows, grid_cols, grid_images);

	dim3 dimGrid(grid_cols, grid_rows, grid_images);
	dim3 dimBlock(block_size, block_size, block_size);

  // const float *image,
  // const float *weight,
  // const float *bias,
  // float *output,
  // const int m,
  // const int n,
  // const int k,
  // const int batch_size,
  // const int appx_mode,
  // const bool use_bias

  matmul_cuda_float<<<dimGrid, dimBlock>>>(
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		m, n, k, b,
    appx_mode,
    use_bias
	);

  cudaDeviceSynchronize();
  return output;
}

torch::Tensor conv_forward_int8(
  torch::Tensor input,
  float input_s,
  int input_zp,
  torch::Tensor weight,
  float weight_s,
  int weight_zp,
  torch::Tensor bias,
	int m,
	int n,
	int k,
	int b,
  int appx_mode
) {

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({b, m, k}, options);

	//std::cout << "Input(CUDA):\n" << input << std::endl;
	//std::cout << "weight(CUDA):\n" << weight << std::endl;

	unsigned int block_size = 8; // Use this block size to not exceed 1024 threads
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;
	unsigned int grid_images = (b + block_size - 1) / block_size;
	//printf("Grid Rows: %d, Grid columns: %d, Grid images: %d\n", grid_rows, grid_cols, grid_images);

	dim3 dimGrid(grid_cols, grid_rows, grid_images);
	dim3 dimBlock(block_size, block_size, block_size);

  // Data is still float type to avoid accumulation issues.
  // for more speed, can perform calculations on char type.

  matmul_cuda_int8<<<dimGrid, dimBlock>>>(
    input.data_ptr<float>(),
    input_s, input_zp,
    weight.data_ptr<float>(),
    weight_s, weight_zp,
    bias.data_ptr<float>(),
    output.data_ptr<float>(),
    m, n, k, b,
    appx_mode
  );

  cudaDeviceSynchronize();
  return output;
}

torch::Tensor linear_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
	int m,
	int n,
	int k,
  int appx_mode,
  bool use_bias
) {

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({m,k}, options);

	unsigned int block_size = 32;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(block_size, block_size);

	matmul_cuda_float<<<dimGrid, dimBlock>>>(
		input.data_ptr<float>(),
		weight.data_ptr<float>(),
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		m, n, k, 1, /* Pass in b=1 since there is no z-dimension for linear layers*/
    appx_mode,
    use_bias
	);

  cudaDeviceSynchronize();
  return output;
}

torch::Tensor linear_forward_int8(
  torch::Tensor input,
  float input_s,
  int input_zp,
  torch::Tensor weight,
  float weight_s,
  int weight_zp,
  torch::Tensor bias,
	int m,
	int n,
	int k,
  int appx_mode
) {

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	auto output = torch::zeros({m,k}, options);

	unsigned int block_size = 32;
	unsigned int grid_rows = (m + block_size - 1) / block_size;
	unsigned int grid_cols = (k + block_size - 1) / block_size;

	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(block_size, block_size);

	matmul_cuda_int8<<<dimGrid, dimBlock>>>(
		input.data_ptr<float>(),
    input_s, input_zp,
		weight.data_ptr<float>(),
    weight_s, weight_zp,
		bias.data_ptr<float>(),
		output.data_ptr<float>(),
		m, n, k, 1, /* Pass in b=1 since there is no z-dimension for linear layers*/
    appx_mode
	);

  cudaDeviceSynchronize();
  return output;
}


PYBIND11_MODULE(cuda_matmul, m) {
  m.def("conv_forward", &conv_forward, "conv_forward (CUDA)");
  m.def("conv_forward_int8", &conv_forward_int8, "conv_forward_int8 (CUDA)");
	m.def("linear_forward", &linear_forward, "linear_forward (CUDA)");
  m.def("linear_forward_int8", &linear_forward_int8, "linear_forward_int8 (CUDA)");
}
