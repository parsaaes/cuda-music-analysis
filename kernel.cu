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
#include <vector>
#include "dirent.h"

using namespace std;

struct Music {
	string address;
	double *powerSpectrum;
};

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
	double * result = (double*) malloc(size * sizeof(double));
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
	result.powerSpectrum = audioFilePowerSpectrum2(address,size);
	return result;
}

int addFiles(vector<struct Music> &vec, char directory[],int size) {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(directory)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (((string)ent->d_name).find(".txt") != std::string::npos) {
				printf("%s\n", ent->d_name);
				vec.push_back(initMusic2(ent->d_name,size));
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
}

int sampleCount(char address[]) {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(address)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (((string)ent->d_name).find(".txt") != std::string::npos) {
				return(numberOfWords(ent->d_name));
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
}

int main()
{
	//double *samplePower = audioFilePowerSpectrum("audio.txt");
	//double *fullPower = audioFilePowerSpectrum("audio2.txt");
	// double *anotherFullPower = audioFilePowerSpectrum("Ehaam-Darya-128.txt");

	/*
	int size = 200;
	int size2 = 100;
	cufftComplex *input = (cufftComplex*)malloc(size * sizeof(cufftComplex));
	cufftComplex *input2 = (cufftComplex*)malloc(size2 * sizeof(cufftComplex));

	for (int i = 0; i < size; i++) {
		input[i].x = 1;
		input[i].y = 0;
	}

	for (int i = 0; i < size2; i++) {
		input2[i].x = 1;
		input2[i].y = 0;
	}

	cufftComplex *output;
	cufftComplex *output2;

	output = (cufftComplex*)malloc(size * sizeof(cufftComplex));
	output2 = (cufftComplex*)malloc(size2 * sizeof(cufftComplex));

	dft(size, input, output);
	dft(size2, input2, output2);

	double * pow1 = powerSpecterum(size, output);

	double * pow2 = powerSpecterum(size2, output2);

	
	printf("%lf", LAD(pow1, pow2, size, size2));
	*/




	char sample_directory[] = "sample\\";
	
	
	vector<struct Music> samples;
	vector<struct Music> full;

	int count = sampleCount(sample_directory);
	addFiles(samples, sample_directory,count);
	addFiles(full, "full\\",count);




	/*
	samples.push_back(initMusic2("1s.txt",numberOfWords("1s.txt")));
	samples.push_back(initMusic2("2s.txt", numberOfWords("2s.txt")));
	samples.push_back(initMusic2("3s.txt", numberOfWords("3s.txt")));
	samples.push_back(initMusic2("4s.txt", numberOfWords("4s.txt")));
	*/
	



	/*
	full.push_back(initMusic2("1.txt", numberOfWords("1s.txt")));
	full.push_back(initMusic2("2.txt", numberOfWords("2s.txt")));
	full.push_back(initMusic2("3.txt", numberOfWords("3s.txt")));
	full.push_back(initMusic2("4.txt", numberOfWords("4s.txt")));
	*/


	for (int i = 0; i < samples.size(); i++) {
		for (int j = 0; j < full.size(); j++) {
			cout << samples[i].address << "-" << full[j].address;
			printf("- %lf\n" ,LAD(samples[i].powerSpectrum, full[j].powerSpectrum, numberOfWords(samples[i].address)));
		}
		printf("\n");
	}

	

	/*
	printf("%lf\n", LAD(samples[0].powerSpectrum, full[0].powerSpectrum, numberOfWords(samples[0].address)));
    printf("%lf\n", LAD(samples[0].powerSpectrum, full[1].powerSpectrum, numberOfWords(samples[0].address)));
	printf("%lf\n\n", LAD(samples[0].powerSpectrum, full[2].powerSpectrum, numberOfWords(samples[0].address)));

	
	printf("%lf\n", LAD(samples[1].powerSpectrum, full[0].powerSpectrum, numberOfWords(samples[1].address)));
	printf("%lf\n", LAD(samples[1].powerSpectrum, full[1].powerSpectrum, numberOfWords(samples[1].address)));
	printf("%lf\n\n", LAD(samples[1].powerSpectrum, full[2].powerSpectrum, numberOfWords(samples[1].address)));

	printf("%lf\n", LAD(samples[2].powerSpectrum, full[0].powerSpectrum, numberOfWords(samples[2].address)));
	printf("%lf\n", LAD(samples[2].powerSpectrum, full[1].powerSpectrum, numberOfWords(samples[2].address)));
	printf("%lf\n\n", LAD(samples[2].powerSpectrum, full[2].powerSpectrum, numberOfWords(samples[2].address)));

	*/

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
