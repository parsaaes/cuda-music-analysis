#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <string.h>
#include <vector>
#include "dirent.h"

using namespace std;

struct Music {
	string address;
	double *powerSpectrum;
};

cudaError_t diff(double *c, const double *a, const double *b, unsigned int size);

__global__ void diffKernel(double *c, const double *a, const double *b,int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		c[i] = a[i] - b[i];
		if (c[i] < 0) {
			c[i] = -c[i];
		}
	}
}

string concat(char x[], char y[]) {
	char result[200];   // array to hold the result.

	strcpy(result, x); // copy string one into the result.
	return strcat(result, y); // append string two to the result.
}

void dft(int size, cufftComplex * input, cufftComplex * output) {
	cufftHandle plan;
	cufftComplex *d_input, *d_output;
	cudaMalloc((void **)&d_input, size * sizeof(cufftComplex));
	cudaMalloc((void **)&d_output, size * sizeof(cufftComplex));
	cudaMemcpy(d_input, input, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cufftPlan1d(&plan, size, CUFFT_C2C, 1);
	cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
}

double* powerSpecterum(int size, cufftComplex * wave) {
	double * result = (double*)malloc(size * sizeof(double));
	//size = size / 2 + 1;
	for (int i = 0; i < size; i++) {
		result[i] = wave[i].x * wave[i].x + wave[i].y * wave[i].y;
	}
	return result;
}

int numberOfWords(string name) {
	int count = 0;
	int x;
	ifstream inFile;
	inFile.open(name);
	if (!inFile) {
		cout << "Unable to open file " << name;
		exit(1); // terminate with error
	}

	while (inFile >> x) {
		count++;
	}
	//cout << count << endl;
	inFile.close();
	return count;
}

cufftComplex* fillFromFile(string name, int size) {
	cufftComplex *input = (cufftComplex*)malloc(size * sizeof(cufftComplex));
	double x;
	ifstream inFile;
	inFile.open(name);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}

	for (int i = 0; i < size; i++) {
		inFile >> x;
		input[i].x = x;
		input[i].y = 0;
	}
	inFile.close();
	return input;
}


double * audioFilePowerSpectrum(string name) {
	cufftComplex *input, *output;
	int fileWordCount = numberOfWords(name);
	input = fillFromFile(name, fileWordCount);
	output = (cufftComplex*)malloc(fileWordCount * sizeof(cufftComplex));
	dft(fileWordCount, input, output);
	return powerSpecterum(fileWordCount, output);
}

double * audioFilePowerSpectrum2(string name, int fileWordCount) {
	cufftComplex *input, *output;
	//int fileWordCount = numberOfWords(name);
	input = fillFromFile(name, fileWordCount);
	output = (cufftComplex*)malloc(fileWordCount * sizeof(cufftComplex));
	dft(fileWordCount, input, output);
	return powerSpecterum(fileWordCount, output);
}

double * audioFilePowerSpectrum3(cufftComplex * input, int fileWordCount) {
	cufftComplex *output;
	output = (cufftComplex*)malloc(fileWordCount * sizeof(cufftComplex));
	dft(fileWordCount, input, output);
	return powerSpecterum(fileWordCount, output);
}

double LAD(double* wave1, double* wave2, int size) {
	//wave1 += size1 / 2;
	//wave2 += size2 / 2;
	int length = (int)fmin(size, 22050);
	double difference = 0;
	for (int i = 0; i < length; i++) {
		difference += fabs(wave2[i] - wave1[i]);
		//printf("%d- %lf %lf\n", i, wave1[i], wave2[i]);
	}
	return difference;
}

double LAD2(double* wave1, double* wave2, int size) {
	int length = (int)fmin(size, 22050);
	double c[22050] = { 0 };
	cudaError_t cudaStatus  = diff(c, wave1, wave2, length);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	double difference = 0;
	for (int i = 0; i < length; i++) {
		difference += c[i];
		//printf("%d- %lf %lf\n", i, wave1[i], wave2[i]);
	}
	return difference;
}

struct Music initMusic(string address) {
	struct Music result;
	result.address = address;
	//result.powerSpectrum = audioFilePowerSpectrum(address);
	result.powerSpectrum = audioFilePowerSpectrum(address);
	return result;
}

struct Music initMusic2(string address, int size) {
	struct Music result;
	result.address = address;
	//result.powerSpectrum = audioFilePowerSpectrum(address);
	result.powerSpectrum = audioFilePowerSpectrum2(address, size);
	return result;
}



int addFiles(vector<struct Music> &vec, char directory[], int size) {
	DIR *dir;
	struct dirent *ent;
	printf("chunk size:%d\n", size);
	if ((dir = opendir(directory)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (((string)ent->d_name).find(".txt") != std::string::npos) {
				cout << concat(directory,ent->d_name) << " ";
				int fullSize = numberOfWords(concat(directory, ent->d_name));
				printf("size:%d\n", fullSize);
				cufftComplex * full = fillFromFile(concat(directory, ent->d_name), fullSize);
				for (int i = 0; i <= fullSize - size; i = i + size) {
					printf("offset:%d-%d\n", i, i + size);
					struct Music aChunk;
					aChunk.address = concat(directory, ent->d_name);
					aChunk.powerSpectrum = audioFilePowerSpectrum3(full + i, size);
					vec.push_back(aChunk);
				}
			}
		}
		printf("\n\n");
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
}

int sampleCount(char address[]) {
	int min = 10000000;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(address)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (((string)ent->d_name).find(".txt") != std::string::npos) {
				int newMin = numberOfWords(concat(address, ent->d_name));
				printf("%d\n", newMin);
				if (min > newMin) {
					min = newMin;
				}
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
	return min;
}


int main(int argc, char **argv)
{
	printf("%s\n", argv[1]);
	printf("%s\n", argv[2]);


	char * sample_directory = argv[2];


	vector<struct Music> samples;
	vector<struct Music> full;

	int count = sampleCount(sample_directory);
	addFiles(samples, sample_directory, count);
	addFiles(full, argv[1], count);




	int minId;
	double minLAD, newLAD;
	for (int i = 0; i < samples.size(); i++) {
		minId = 0;
		minLAD = LAD2(samples[i].powerSpectrum, full[0].powerSpectrum, numberOfWords(samples[i].address));
		for (int j = 1; j < full.size(); j++) {
			newLAD = LAD2(samples[i].powerSpectrum, full[j].powerSpectrum, numberOfWords(samples[i].address));
			cout << samples[i].address << "-" << full[j].address;
			printf(": %lf\n",newLAD);
			if (newLAD < minLAD) {
				minId = j;
				minLAD = newLAD;
			}
		}
		printf("\n");
		cout << samples[i].address << " >>> " << full[minId].address;
		printf("\n");

	}
	

	/*
	const int arraySize = 5;
	double a[arraySize] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
	double b[arraySize] = { 10.0, 20.0, 30.0, 40.0, 50.0 };
	double c[arraySize] = { 0 };

	printf("%lf", LAD2(a, b, 5));
	*/


	


	return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t diff(double *c, const double *a, const double *b, unsigned int size)
{
	double *dev_a = 0;
	double *dev_b = 0;
	double *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	diffKernel << <ceil(size/1024.0), 1024 >> >(dev_c, dev_a, dev_b,(int)size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

