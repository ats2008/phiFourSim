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


__global__ void checkBoardUpdate(float* latticeArray,short mode,float tempAssignNumber, int tStepCount_, const int NTot)
{
	//assert(mode==0 or mode==1);
	int tidX =  2*(blockDim.x * blockIdx.x + threadIdx.x) + mode;
	int tidY =  2*(blockDim.y * blockIdx.y + threadIdx.y) + mode;
	int tidZ =  2*(blockDim.z * blockIdx.z + threadIdx.z) + mode;
	
	int xyzPos = tidX * gridDim.y*blockDim.y * gridDim.z*blockDim.z + tidY * gridDim.z*blockDim.z+tidZ;

	auto gridOffset =gridDim.x*blockDim.x * gridDim.y*blockDim.y * gridDim.z*blockDim.z;
	

	int tId=mode;
	while(tId<tStepCount_)
	{
		
		auto posIdx= tId*gridOffset + xyzPos ;
		printf("grid offset = %d, posidx =%d tID = %d , xyzPos = %d [%d,%d,%d] \n ",gridOffset,posIdx,tId,xyzPos,tidX,tidY,tidZ);
		
		//assert(posIdx < NTot );
		latticeArray[posIdx]=tempAssignNumber;
		
	tId+=2; 	
	}
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

void phiFourLattice::doGPUlatticeUpdates( int numUpdates)
{
	dim3 gridSize(2,2,2);
	dim3 blockSize(2,2,2);

	std::cout<<"\n Launching the kerrnels for Lattice Size = "<<latticeSize_<<" ( t_d = "<<tStepCount_<<" x_d = "<<xStepCount_<<" & D = "<<dim_<<"\n";
	checkBoardUpdate<<<gridSize,blockSize,0>>>( CurrentStateGPU , 0 , 1.0,tStepCount_ ,latticeSize_ );
	checkBoardUpdate<<<gridSize,blockSize,0>>>( CurrentStateGPU , 1 , 2.0,tStepCount_ ,latticeSize_ );
	cudaThreadSynchronize();
}


