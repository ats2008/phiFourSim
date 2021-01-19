#include "lattice.h"
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void testrun()
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	printf("tId : %d \n",tId);
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


__global__ void checkBoardUpdate(float* latticeArray,int mode,float tempAssignNumber, int tStepCount_, const int NTot)
{
	//assert(mode==0 or mode==1);
	int tIdX =  (blockDim.x * blockIdx.x + threadIdx.x) ;
	int tIdY =  (blockDim.y * blockIdx.y + threadIdx.y) ;
	int tIdZ =  (blockDim.z * blockIdx.z + threadIdx.z) ;
	
	int  xyzPos     = tIdX *gridDim.y*blockDim.y * gridDim.z*blockDim.z + tIdY * gridDim.z*blockDim.z+tIdZ;
	
	auto gridOffset = gridDim.x*blockDim.x * gridDim.y*blockDim.y * gridDim.z*blockDim.z;
	
	auto tId        = (tIdX + tIdY + tIdZ )%2 ;
	
	if( mode==1 && tId==0 ) tId=1;
	if( mode==1 && tId==1 ) tId=0;
	// if(mode==0 && tId==0 ) tId=0;
	// if(mode==0 && tId==1 ) tId=1;
	
	while(tId<tStepCount_)
	{
		
		auto posIdx= tId*gridOffset + xyzPos ;
		latticeArray[posIdx]=tempAssignNumber;
		
		printf("grid offset = %d, posidx =%d tID = %d , xyzPos = %d [ tIdX %d, tIdY %d, tIdZ %d ] [ bidX:%d, bidY:%d, bidZ:%d ] latticeArray[%d] -> %f \n ",
						gridOffset,posIdx,tId,xyzPos,tIdX,tIdY,tIdZ,blockIdx.x,blockIdx.y,blockIdx.z,posIdx,latticeArray[posIdx]);
		
		//assert(posIdx < NTot );
		
	 	tId+=2; 	
	}
}

void phiFourLattice::phiFourLatticeGPUConstructor()
{
	cudaMalloc(&CurrentObservablesGPU,latticeSize_);
	cudaMalloc(&CurrentStateGPU,latticeSize_*sizeof(float));
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

	cudaDeviceSynchronize();
}
void phiFourLattice::initializeLatticeGPU()
{
	return;
}

void phiFourLattice::copyStateInCPUtoGPU()
{
	std::cout<<"\n Copying to the GPU \n";
	cudaMemcpy(CurrentStateGPU,CurrentStateCPU,latticeSize_*sizeof(float),cudaMemcpyHostToDevice); 	
}
void phiFourLattice::copyStateInGPUtoCPU()
{
	cout<<"\n\n latticeSize_*sizeof(float) = "<<latticeSize_<<" * "<<sizeof(float)<<"\n\n";
	cudaMemcpy(CurrentStateCPU,CurrentStateGPU,latticeSize_*sizeof(float),cudaMemcpyDeviceToHost); 	
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
	cudaDeviceSynchronize();
}

void phiFourLattice::doGPUlatticeUpdates( int numUpdates)
{
	const int blockLen=4 ; //8 ;
	const int gridLen=( xStepCount_/blockLen );
	dim3 blockSize(blockLen,blockLen,blockLen);
	dim3 gridSize(gridLen,gridLen,gridLen);
 
	std::cout<<"Launching the kerrnels for Lattice Size = "<<latticeSize_<<" ( t_d = "<<tStepCount_<<" x_d = "<<xStepCount_<<" & D = "<<dim_<<"\n"
		 <<" with grid size : "<<gridSize.x<<" , "<<gridSize.y<<" , "<<gridSize.z<<"\n"
		 <<" and block size : "<<blockSize.x<<" , "<<blockSize.y<<" , "<<blockSize.z<<"\n";	
	checkBoardUpdate<<<gridSize,blockSize>>>( CurrentStateGPU , 0 , 1.0,tStepCount_ ,latticeSize_ );
	cudaThreadSynchronize();
	checkBoardUpdate<<<gridSize,blockSize>>>( CurrentStateGPU , 1 , 2.0,tStepCount_ ,latticeSize_ );
	cudaDeviceSynchronize();
}


