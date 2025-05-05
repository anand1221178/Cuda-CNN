// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check
#define MAX_EPSILON_ERROR 5e-3f
const char *sampleName = "simpleTexture";
// Define the files that are to be save and the reference images for validation
const char *imageFilename = "teapot512.pgm";

// Define the size of the convolution mask
const int mask_size = 3;

// Forward declarations
float runSerial(int argc, char **argv);
float runGlobalCuda(int argc, char **argv);

// Serial convolution implementation
__host__ void serialConvolution(float* input, float* output, int width, int height, const int* mask, int mask_size)
{
	//1) find a and b -> in this case a=b since l x l images
	int radius_lp = mask_size/2;

	// 2) Center kernal F at pixel (i,j) in image 
	
	// Iterate over rows first since we are using row major layuout the i will jump ""width"" elements to the next row where the j will iterate through each element in the "row" 
	for(int i = 0; i < height; i++) //iterate over y axis -> height
	{
		for(int j = 0; j < width; j++) //iterate over x axis -> width
		{
			// Declare sum for this pixel
			double sum = 0.0;
			// Loop over kernal values l and p
			for(int l = -radius_lp; l <= radius_lp; ++l)
			{
				for(int p = -radius_lp; p <= radius_lp; ++p)
				{
					// Get pixel coords of input point
					int image_x = j + p; //move along cols
					int image_y = i + l; //move along rowss					

					// Check if pixel is in bound or not
					if(image_x >= 0 && image_x < width && image_y >= 0 && image_y < height) //) padding, outside pixels are 0
					{
						int maskIndex = (l + radius_lp) * mask_size + (p + radius_lp); //Since we store the mask as row vector 
						float imagePixel = input[image_y * width + image_x]; //Simple idx'ng for 2D row major access
						int kernelWeight = mask[maskIndex];
						sum += imagePixel * kernelWeight;

					}
				}
			}
			if(sum < 0) sum =0;
			if(sum > 255) sum = 255;
			output[i * width + j] = sum;
		}
	}
}

// Cuda global memory implementation
__global__ void globalMemconv(float* d_input_image, float* d_output_image, int width, int height, float* mask, int mask_size)
{
	// Since we are using row major ordering
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x; //cols (X)
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y; //rows (y)

	// INitial in bounds check
	if (idx_x < width && idx_y < height)
	{
		// Do convolution for a single thread

		// Mask radius
		int radius_lp = mask_size/2;

		// Accumalation sum
		double sum = 0.0;

		for(int l = -radius_lp; l <= radius_lp; ++l)
			{
				for(int p = -radius_lp; p <= radius_lp; ++p)
				{
					// Get pixel coords of input point
					int image_x = idx_x + p; //move along cols
					int image_y = idx_y + l; //move along rowss					

					// Check if pixel is in bound or not
					if(image_x >= 0 && image_x < width && image_y >= 0 && image_y < height) //) padding, outside pixels are 0
					{
						int maskIndex = (l + radius_lp) * mask_size + (p + radius_lp); //Since we store the mask as row vector 
						float imagePixel = d_input_image[image_y * width + image_x]; //Simple idx'ng for 2D row major access
						float kernelWeight = mask[maskIndex];
						sum += imagePixel * kernelWeight;
					}
				}
			}
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;

			d_output_image[idx_y * width + idx_x] = sum; //Row major storing method

	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	printf("%s starting...\n", sampleName);

	// Process command-line arguments
	if (argc > 1) {
	if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
		getCmdLineArgumentString(argc, (const char **)argv, "input",
								(char **)&imageFilename);
	} else if (checkCmdLineFlag(argc, (const char **)argv, "reference")) {
		printf("-reference flag should be used with -input flag");
		exit(EXIT_FAILURE);
	}
	}

	float milliseconds_serial, milliseconds_cuda_global;

	//Run the serial version
	milliseconds_serial = runSerial(argc, argv);

	milliseconds_cuda_global = runGlobalCuda(argc, argv);

	printf("Time taken serially: %f\n", milliseconds_serial);
	printf("Time taken cuda globally : %f\n", milliseconds_cuda_global);

	printf("Speedup: serial vs cuda global memory: %fx\n", milliseconds_serial/milliseconds_cuda_global);

	return 0;

}

float runGlobalCuda(int argc, char **argv){
	// Define host input and output 
	float *h_input_image = NULL;
	float *h_output_image = NULL;
	float *d_input_image = NULL;
	float *d_output_image = NULL;

	unsigned int width, height;

	char *image_path = sdkFindFilePath(imageFilename, argv[0]);

	if(image_path == NULL)
	{
		printf("Unable to find image file: %s\n", imageFilename);
		exit(EXIT_FAILURE);
	}
	
	sdkLoadPGM(image_path, &h_input_image, &width, &height);

	// Allocate host output
	h_output_image = (float *)malloc(sizeof(float) * width * height);

	// Allocate device mem
	checkCudaErrors(cudaMalloc((void **)&d_input_image, sizeof(float)* width * height));
	checkCudaErrors(cudaMalloc((void **)&d_output_image, sizeof(float) * width * height));

	// Copy input image from HOST to Device
	checkCudaErrors(cudaMemcpy(d_input_image, h_input_image, sizeof(float) * height * width, cudaMemcpyHostToDevice));


	const float sharpen_mask[9] = {-1.0f,-1.0f,-1.0f,
								-1.0f,9.0f,-1.0f,
								-1.0f,-1.0f,-1.0f}; 

	float *d_mask = NULL;
	checkCudaErrors(cudaMalloc((void **)&d_mask, sizeof(float) * mask_size * mask_size));
	checkCudaErrors(cudaMemcpy(d_mask, sharpen_mask, sizeof(float) * mask_size * mask_size, cudaMemcpyHostToDevice));
	// Kernal implementation -> for now assume 512 x 512 image
	//!In future check how many SMs check how many blocks per sm and how many threads per SM then allocate
	int devCount;
	cudaGetDeviceCount(&devCount);

	int selectedDevice = -1;
	cudaDeviceProp devProp;

	// Find non integrated graphics card to use
	for (int i = 0; i < devCount; ++i) {
		cudaGetDeviceProperties(&devProp, i);

		if (!devProp.integrated) {
			selectedDevice = i;
			break;
		}
	}

	// Error return early 
	if (selectedDevice == -1) {
		fprintf(stderr, "No suitable CUDA device found!\n");
		exit(EXIT_FAILURE);
	}

	// Set to nvidia GPU
	cudaSetDevice(selectedDevice);
	cudaGetDeviceProperties(&devProp, selectedDevice);

	printf("Using GPU: %s\n", devProp.name);
	printf("Max threads per block: %d\n", devProp.maxThreadsPerBlock);
	printf("Max thread dimensions: %d x %d x %d\n", 
			devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
	printf("Max grid size: %d x %d x %d\n",
			devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);


	// Launch 32 blocks with 32 threads each block to cover entire image;
	dim3 threads_per_block(32, 32);  // 256 threads
	dim3 n_blocks(
		(width  + threads_per_block.x - 1) / threads_per_block.x,
		(height + threads_per_block.y - 1) / threads_per_block.y
	);


	cudaEvent_t start_global, stop_global;
	cudaEventCreate(&start_global);
	cudaEventCreate(&stop_global);

	cudaEventRecord(start_global);
	// Kernal launch
	globalMemconv<<<n_blocks, threads_per_block>>>(d_input_image, d_output_image, width, height, d_mask, mask_size);
	cudaEventRecord(stop_global);
	// Copy result back to host mem from device memory
	checkCudaErrors(cudaMemcpy(h_output_image, d_output_image, sizeof(float) * height * width, cudaMemcpyDeviceToHost));

	// Save output
	sdkSavePGM("global_output.pgm", h_output_image, width, height);
	printf("Saved global_output.pgm\n");

	float milliseconds_global = 0;
	cudaEventElapsedTime(&milliseconds_global, start_global, stop_global);

	

	// Freeup
	free(h_output_image);
	cudaFree(d_input_image);
	cudaFree(d_output_image);
	free(image_path);

	return milliseconds_global;
}

float runSerial(int argc, char **argv) {
  // LOAD IMAGE
  
	float *input_image = NULL;
	float *output_image = NULL;

	//Define image dims
	unsigned int width, height; 

	// IMAGE PATH
	char *image_path = sdkFindFilePath(imageFilename, argv[0]);
	
	//Check to see if image path has been provided
	if(image_path == NULL)
	{
		printf("Unable to find image file: %s\n", imageFilename);
		exit(EXIT_FAILURE);
	}

	sdkLoadPGM(image_path, &input_image, &width, &height);

	// Define output image size of type float of width times height * size of single float
	output_image = (float * )malloc(width * height * sizeof(float));

	// example mask: has to be linear since we store in row major order
	const int sharpen_mask[9] = {-1,-1,-1,
								-1,9,-1,
								-1,-1,-1}; 
	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start);
	// Host function since we are running it on the CPU 
	serialConvolution(input_image, output_image, width, height, sharpen_mask, mask_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	sdkSavePGM("serial_output.pgm", output_image, width, height);

	printf("Saved: serial_output.pgm\n");

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	

	free(output_image);
	free(image_path);

	return milliseconds;
}