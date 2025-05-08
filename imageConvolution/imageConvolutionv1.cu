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

// Filters
enum FilterType { SHARPEN = 0, AVERAGE = 1, EMBOSS = 2 };

//define masks
// 3x3
const float sharpen3x3[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
const float average3x3[9] = {1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f};
const float emboss3x3[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};

// 5x5
const float sharpen5x5[25] = {-1, -1, -1, -1, -1,
								-1,  -1,  -1,  -1, -1,
								-1,  -1,  9,  -1, -1,
								-1,  -1,  -1,  -1, -1,
								-1, -1, -1, -1, -1};

const float average5x5[25] = { 1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f,
								1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f,
								1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f,
								1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f,
								1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f, 1/25.0f};

const float emboss5x5[25] = {
						1, 0, 0, 0, 0,
						0, 1, 0, 0, 0,
						0, 0, 0, 0, 0,
						0, 0, 0,-1, 0,
						0, 0, 0, 0,-1
					};

// 7x7
const float sharpen7x7[49] = {-1, -1, -1, -1, -1, -1, -1,
								-1,  -1,  -1,  -1,  -1,  -1,  -1,
								-1,  -1,  -1,  -1,  -1,  -1,  -1,
								-1,  -1,  -1, 9.0f,  -1,  -1,  -1,
								-1,  -1,  -1,  -1,  -1,  -1,  -1,
								-1,  -1,  -1,  -1,  -1,  -1,  -1,
								-1,-1,-1,-1,-1,-1,-1};

const float average7x7[49] = {1/49.0f , 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
								1/49.0f , 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
								1/49.0f , 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
								1/49.0f , 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
								1/49.0f , 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
								1/49.0f , 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f,
								1/49.0f , 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f, 1/49.0f};
const float emboss7x7[49] = {
						1, 0, 0, 0, 0, 0, 0,
						0, 1, 0, 0, 0, 0, 0,
						0, 0, 1, 0, 0, 0, 0,
						0, 0, 0,0, 0, 0, 0,
						0, 0, 0, 0,-1, 0, 0,
						0, 0, 0, 0, 0,-1, 0,
						0, 0, 0, 0, 0, 0, -1
					};


// Forward declarations
float runSerial(int argc, char **argv, const float *selected_filter_type, int selected_mask_size);
float runGlobalCuda(int argc, char **argv, const float *selected_filter_type, int selected_mask_size);
float runSharedCuda(int argc, char **argv, const float *selected_filter_type, int selected_mask_size);

// Serial convolution implementation
__host__ void serialConvolution(float* input, float* output, int width, int height, const float* mask, int mask_size){
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
						float kernelWeight = mask[maskIndex];
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
__global__ void globalMemConv(float* d_input_image, float* d_output_image, int width, int height, float* mask, int mask_size){
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

// Cuda shared memory implementation
__global__ void sharedMemConv(float* input, float* output, int width, int height, const float* mask, int mask_size){

	// Setup shared memory for this tile of mask size, + thread block size
	extern __shared__ float tile[];

	// Thread x and y indexes within block
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Row major thread indexing nfor absolute position in image
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Mask size used to deine padding around tile
	int radius = mask_size/2;
	
	// Shared memory within the thread block is the thread block itself + the radius of the mask;
	int shared_width = blockDim.x + 2 * radius;

	// Index of shared threads in tiles + offset by radius since we want the center pixel but we have padding on the edges
	int shared_x = tx + radius;
	int shared_y = ty + radius;

	// Load center pixel of each thread into shared memory
	// Check if in bounds
	if(x < width && y < height)
	{
		// Load into shared memory -> remember y * width + x for indexing in row major
		tile[shared_y * shared_width + shared_x] = input[y * width + x];
	}
	else{
		// Set to 0 else
		tile[shared_y * shared_width + shared_x] = 0.0f;
	}


	if (tx < radius) {
		// left border
		int img_x = x - radius;
		tile[shared_y * shared_width + tx] = (img_x >= 0 && y < height)
											  ? input[y * width + img_x] : 0.0f;
	
		// right border
		img_x = x + blockDim.x;
		tile[shared_y * shared_width + (tx + blockDim.x + radius)] =
			(img_x < width && y < height) ? input[y * width + img_x] : 0.0f;
	}
	
	if (ty < radius) {
		// top border
		int img_y = y - radius;
		tile[ty * shared_width + shared_x] = (x < width && img_y >= 0)
											 ? input[img_y * width + x] : 0.0f;
	
		// bottom border
		img_y = y + blockDim.y;
		tile[(ty + blockDim.y + radius) * shared_width + shared_x] =
			(x < width && img_y < height) ? input[img_y * width + x] : 0.0f;
	}
	if (tx < radius && ty < radius) {
		// top-left
		int img_x = x - radius;
		int img_y = y - radius;
		tile[ty * shared_width + tx] = (img_x >= 0 && img_y >= 0) ? input[img_y * width + img_x] : 0.0f;
	
		// top-right
		img_x = x + blockDim.x;
		img_y = y - radius;
		tile[ty * shared_width + (tx + blockDim.x + radius)] = (img_x < width && img_y >= 0) ? input[img_y * width + img_x] : 0.0f;
	
		// bottom-left
		img_x = x - radius;
		img_y = y + blockDim.y;
		tile[(ty + blockDim.y + radius) * shared_width + tx] = (img_x >= 0 && img_y < height) ? input[img_y * width + img_x] : 0.0f;
	
		// bottom-right
		img_x = x + blockDim.x;
		img_y = y + blockDim.y;
		tile[(ty + blockDim.y + radius) * shared_width + (tx + blockDim.x + radius)] =
			(img_x < width && img_y < height) ? input[img_y * width + img_x] : 0.0f;
	}
	

	__syncthreads();
	// Now do convolution
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            for (int j = -radius; j <= radius; ++j) {
                float pixel = tile[(shared_y + i) * shared_width + (shared_x + j)];
                float weight = mask[(i + radius) * mask_size + (j + radius)];
                sum += pixel * weight;
            }
        }
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        output[y * width + x] = sum;
    }

}

// Mask selector
const float* getMask(FilterType filter, int mask_size) {
    switch (mask_size) {
        case 3:
            if (filter == SHARPEN) return sharpen3x3;
            if (filter == AVERAGE) return average3x3;
            return emboss3x3;
        case 5:
            if (filter == SHARPEN) return sharpen5x5;
            if (filter == AVERAGE) return average5x5;
            return emboss5x5;
        case 7:
            if (filter == SHARPEN) return sharpen7x7;
            if (filter == AVERAGE) return average7x7;
            return emboss7x7;
        default:
            fprintf(stderr, "Unsupported mask size\n");
            exit(EXIT_FAILURE);
    }
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
	printf("%s starting...\n", sampleName);

	int mask_size = 3;
    FilterType filter = SHARPEN;


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
	// Set mask size
	if (checkCmdLineFlag(argc, (const char **)argv, "mask")) {
        mask_size = getCmdLineArgumentInt(argc, (const char **)argv, "mask");
        if (mask_size != 3 && mask_size != 5 && mask_size != 7) {
            fprintf(stderr, "Invalid mask size. Use -mask=3, 5 or 7\n");
            return EXIT_FAILURE;
        }
    }
	// Set filter type
    if (checkCmdLineFlag(argc, (const char **)argv, "filter")) {
        char *filter_arg;
        getCmdLineArgumentString(argc, (const char **)argv, "filter", &filter_arg);
        if (strcmp(filter_arg, "sharpen") == 0) {
            filter = SHARPEN;
        } else if (strcmp(filter_arg, "average") == 0) {
            filter = AVERAGE;
        } else if (strcmp(filter_arg, "emboss") == 0) {
            filter = EMBOSS;
        } else {
            fprintf(stderr, "Invalid filter type. Use -filter=sharpen, average or emboss\n");
            return EXIT_FAILURE;
        }
    }

	const float* selected_mask = getMask(filter, mask_size);

	float t_serial = runSerial(argc, argv, selected_mask, mask_size);
    float t_global = runGlobalCuda(argc, argv, selected_mask, mask_size);
    float t_shared = runSharedCuda(argc, argv, selected_mask, mask_size);

	// Print out the results
	printf("============================================\n");
	printf("Timings:\n");
	printf("Time taken serially: %f\n", t_serial);
	printf("Time taken cuda globally : %f\n", t_global);
	printf("Time take cuda shared : %f\n", t_shared);


	printf("============================================\n");
	printf("Speedups:\n");
	printf("Speedup: serial vs cuda global memory: %fx\n", t_serial/t_global);
	printf("Speedup: serial vs cuda shared memory: %fx\n", t_serial/t_shared);
	printf("Speedup: cuda global memory vs cuda shared memory: %fx\n", t_global/t_shared);
	printf("============================================\n");
	return 0;

}

float runSharedCuda(int argc, char **argv, const float *selected_mask, int mask_size){

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

	float *d_mask = NULL;
	checkCudaErrors(cudaMalloc((void **)&d_mask, sizeof(float) * mask_size * mask_size));
	checkCudaErrors(cudaMemcpy(d_mask, selected_mask, sizeof(float) * mask_size * mask_size, cudaMemcpyHostToDevice));

	// Select appropriate CUDA device
	int devCount;
	cudaGetDeviceCount(&devCount);

	int selectedDevice = -1;
	cudaDeviceProp devProp;

	for (int i = 0; i < devCount; ++i) {
		cudaGetDeviceProperties(&devProp, i);
		if (!devProp.integrated) {
			selectedDevice = i;
			break;
		}
	}
	if (selectedDevice == -1) {
		fprintf(stderr, "No suitable CUDA device found!\n");
		exit(EXIT_FAILURE);
	}

	cudaSetDevice(selectedDevice);
	cudaGetDeviceProperties(&devProp, selectedDevice);

	printf("Using GPU: %s\n", devProp.name);
	printf("Max threads per block: %d\n", devProp.maxThreadsPerBlock);
	printf("Max thread dimensions: %d x %d x %d\n", 
			devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
	printf("Max grid size: %d x %d x %d\n",
			devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);

	// Launch configuration
	dim3 threads_per_block(16, 16);
	dim3 n_blocks(
		(width  + threads_per_block.x - 1) / threads_per_block.x,
		(height + threads_per_block.y - 1) / threads_per_block.y
	);

	// Create CUDA timers
	cudaEvent_t start_shared, stop_shared;
	cudaEventCreate(&start_shared);
	cudaEventCreate(&stop_shared);

	//Also pass the shared mem size
	int radius = mask_size / 2;
	int shared_mem_size = (threads_per_block.x + 2 * radius) * (threads_per_block.y + 2 * radius) * sizeof(float);

	// Launch the kernel
	cudaEventRecord(start_shared);
	sharedMemConv<<<n_blocks, threads_per_block, shared_mem_size>>>(
		d_input_image,
		d_output_image,
		width,
		height,
		d_mask,
		mask_size
	);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_shared);
	cudaEventSynchronize(stop_shared);

	// Copy result back to host
	checkCudaErrors(cudaMemcpy(h_output_image, d_output_image, sizeof(float) * height * width, cudaMemcpyDeviceToHost));

	// Save output
	sdkSavePGM("shared_output.pgm", h_output_image, width, height);
	printf("Saved shared_output.pgm\n");

	// Report timing
	float milliseconds_shared = 0;
	cudaEventSynchronize(stop_shared);
	cudaEventElapsedTime(&milliseconds_shared, start_shared, stop_shared);

	// Cleanup
	free(h_output_image);
	cudaFree(d_input_image);
	cudaFree(d_output_image);
	cudaFree(d_mask);
	free(image_path);

	return milliseconds_shared;
}



float runGlobalCuda(int argc, char **argv, const float *selected_mask, int mask_size){
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
	// copy mask
	float *d_mask = NULL;
	checkCudaErrors(cudaMalloc((void **)&d_mask, sizeof(float) * mask_size * mask_size));
	checkCudaErrors(cudaMemcpy(d_mask, selected_mask, sizeof(float) * mask_size * mask_size, cudaMemcpyHostToDevice));
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
	globalMemConv<<<n_blocks, threads_per_block>>>(d_input_image, d_output_image, width, height, d_mask, mask_size);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_global);
	cudaEventSynchronize(stop_global);
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

float runSerial(int argc, char **argv, const float *selected_mask, int mask_size){
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

	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start);
	// Host function since we are running it on the CPU 
	serialConvolution(input_image, output_image, width, height, selected_mask, mask_size);
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