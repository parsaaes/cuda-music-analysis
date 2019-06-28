#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

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

int numberOfWords(char name[]) {
	int count = 0;
	int x;
	ifstream inFile;
	inFile.open(name);
	if (!inFile) {
		cout << "Unable to open file";
		exit(1); // terminate with error
	}

	while (inFile >> x) {
		count++;
	}
	//cout << count << endl;
	inFile.close();
	return count;
}

cufftComplex* fillFromFile(char name[], int size) {
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
	return input;
}

double * audioFilePowerSpectrum(char name[]) {
	cufftComplex *input, *output;
	int fileWordCount = numberOfWords(name);
	input = fillFromFile(name, fileWordCount);
	output = (cufftComplex*)malloc(fileWordCount * sizeof(cufftComplex));
	dft(fileWordCount, input, output);
	return powerSpecterum(fileWordCount, output);
}

double LAD(double* wave1, double* wave2, int size) {
	double difference = 0;
	for (int i = 0; i < size; i++) {
		difference += fabs(wave2[i] - wave1[i]);
	}
	return difference;
}

int main()
{
	double *samplePower = audioFilePowerSpectrum("audio.txt");
	double *fullPower = audioFilePowerSpectrum("audio2.txt");

	for (int i = 0; i < 10; i++) {
		printf("%lf \n", samplePower[i]);
		printf("%lf \n\n", fullPower[i]);
	}
	printf("%lf\n", LAD(samplePower, fullPower, numberOfWords("audio.txt")));
	printf("%lf\n", LAD(samplePower, samplePower, numberOfWords("audio.txt")));


	/*
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
	*/

    return 0;
}
