#include "lattice.h"
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void testrun()
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	printf("tid : %d \n",tid);
	return;
}

__global__ void printLattice(float * lattice,uint16_t l1,uint16_t lTotal)
{
	int idx=blockIdx.x*l1+threadIdx.x;
	if(idx<lTotal)
	{
		printf("blockIdx.x*l1 + threadIdx.x = %d*%d+%d = %d -> %f \n",
					blockIdx.x,l1,threadIdx.x,blockIdx.x*l1+threadIdx.x,lattice[idx]);
	}
	return;
}

void phiFourLattice::phiFourLatticeGPUConstructor()
{
	cudaMalloc(&CurrentObservablesGPU,latticeSize_);
	cudaMalloc(&CurrentStateGPU,latticeSize_);
}
void phiFourLattice::phiFourLatticeGPUDistructor()
{
	cudaFree(CurrentStateGPU);
	cudaFree(CurrentObservablesGPU);
}

void phiFourLattice::simplePrintfFromKernel()
{

	printf("haha in the wraper \n");
	testrun<<<5,2>>>();
	printf("haha leaving the wraper \n");

	cudaThreadSynchronize();
}
void phiFourLattice::initializeLatticeGPU()
{
	return;
}

void phiFourLattice::copyStateInCPUtoGPU()
{
	std::cout<<"\n Copying to the GPU \n";
	cudaMemcpy(CurrentStateGPU,CurrentStateCPU,latticeSize_,cudaMemcpyHostToDevice); 	
}
void phiFourLattice::copyStateToGPUtoCPU()
{
	cudaMemcpy(CurrentStateGPU,CurrentStateCPU,latticeSize_,cudaMemcpyDeviceToHost); 	
}
void phiFourLattice::copyObservalblesInGPUToaCPU()
{
	cudaMemcpy(CurrentObservablesCPU,CurrentObservablesGPU,5,cudaMemcpyHostToDevice); 	
}
void phiFourLattice::copyObservalblesInCPUToGPU()
{
	cudaMemcpy(CurrentObservablesGPU,CurrentObservablesCPU,5,cudaMemcpyDeviceToHost); 	
}

void phiFourLattice::printLatticeOnGPU()
{
	int numberOfBlocks=latticeSize_/tStepCount_ + 1;
	int threadsPerBlock=tStepCount_;
	printLattice<<<numberOfBlocks,threadsPerBlock  >>>(CurrentStateGPU,xStepCount_,latticeSize_);
	cudaThreadSynchronize();
}


