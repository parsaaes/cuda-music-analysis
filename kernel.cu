
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include <math.h>


void dft(int size, cufftComplex * input, cufftComplex * output) {
	cufftHandle plan;
	cufftComplex *d_input, *d_output;
	cudaMalloc((void **)&d_input, size * sizeof(cufftComplex));
	cudaMalloc((void **)&d_output, size * sizeof(cufftComplex));
	cudaMemcpy(d_input, input, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cufftPlan1d(&plan, size, CUFFT_C2C, 1);
	cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
}

double* powerSpecterum(int size, cufftComplex * wave) {
	double * result = (double*) malloc(size * sizeof(double));
	for (int i = 0; i < size; i++) {
		result[i] = wave[i].x * wave[i].x + wave[i].y * wave[i].y;
	}
	return result;
}

int main()
{
	cudaSetDevice(0);
	cufftComplex *input, *output;
	int size = 100;
	input = (cufftComplex*)malloc(size * sizeof(cufftComplex));
	output = (cufftComplex*)malloc(size * sizeof(cufftComplex));

	for (int i = 0; i < size; i++) {
		input[i].x = (float)cos(3.14 + 3.14 * i);
		input[i].y = 0;
	}

	dft(size, input, output);
	double* power = powerSpecterum(size, output);

	for (int i = 0; i < size; i++) {
		printf("%d- %f^2 + %f^2 = %f \n",i,output[i].x, output[i].y, power[i]);
	}

    return 0;
}
