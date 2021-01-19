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

__device__ int  getNeighbour(int probeDim,int dir, int tId,int tStepCount_,int xyzblockSize)
{
	if(probeDim == 1 ) 
	{
		auto neibBlkIdx = (blockIdx.x + dir < 0 or blockIdx.x + dir > gridDim.x -1 )? (blockIdx.x + dir + gridDim.x )%gridDim.x : blockIdx.x;
		auto neibThrIdx = (threadIdx.x + dir + blockDim.x )%blockDim.x;
		
		auto xyzblockNumber = neibBlkIdx*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
		auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ neibThrIdx*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +threadIdx.z ;
		auto posIdx= tId*xyzblockSize + xyzPos ;
		
		return posIdx;
	}
	if(probeDim == 2 ) 
	{
		auto neibBlkIdx = (blockIdx.y +dir < 0 or blockIdx.y + dir > gridDim.y -1 )? (blockIdx.y + dir + gridDim.y )%gridDim.y : blockIdx.y;
		auto neibThrIdx = (threadIdx.y+ dir + blockDim.y )%blockDim.y;
		
		auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + neibBlkIdx*gridDim.z + blockIdx.z;
		auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + neibThrIdx*blockDim.z +threadIdx.z ;
		auto posIdx= tId*xyzblockSize + xyzPos ;
		
		return posIdx;
	}
	if(probeDim == 3 ) 
	{
		auto neibBlkIdx = (blockIdx.z +dir < 0 or blockIdx.z + dir > gridDim.z -1 )? (blockIdx.z + dir + gridDim.z )%gridDim.z : blockIdx.z;
		auto neibThrIdx = (threadIdx.z + dir + blockDim.z )%blockDim.z;
		
		auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + neibBlkIdx;
		auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +neibThrIdx ;
		auto posIdx= tId*xyzblockSize + xyzPos ;
		
		return posIdx;
	}
	
	return -1;
}

__global__ void checkBoardUpdate(float* latticeArray,int mode,float tempAssignNumber, int tStepCount_, const int NTot)
{
	//assert(mode==0 or mode==1);
	int tIdX =  ( threadIdx.x) ;
	int tIdY =  ( threadIdx.y) ;
	int tIdZ =  ( threadIdx.z) ;
	
//	int  xyzPos     = tIdX *gridDim.y*blockDim.y * gridDim.z*blockDim.z + tIdY * gridDim.z*blockDim.z+tIdZ;
	
	auto xyzblockSize   = blockDim.x*blockDim.y*blockDim.z;
	auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
	auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +threadIdx.z ;
		
	//auto gridOffset   = gridDim.x*blockDim.x * gridDim.y*blockDim.y * gridDim.z*blockDim.z;
	auto tId = (threadIdx.x + threadIdx.y + threadIdx.z ) % 2 ;
	
	if( mode==1 && tId==0 ) tId=1;
	else if( mode==1 && tId==1 ) tId=0;
	
	// if(mode==0 && tId==0 ) tId=0;
	// if(mode==0 && tId==1 ) tId=1;
	
	while(tId<tStepCount_)
	{
		
		auto posIdx= tId*xyzblockSize + xyzPos ;
		latticeArray[posIdx]=tempAssignNumber;
		
		auto neib=( (tId+1 + tStepCount_)%tStepCount_)*xyzblockSize + xyzPos  ;
		//printf("neib : %f , ",neib);
		latticeArray[neib]=-1*tempAssignNumber;
		auto neibA = getNeighbour(1, 1 , tId,tStepCount_,xyzblockSize);
		latticeArray[neibA] = -1*tempAssignNumber;
		auto neibB = getNeighbour(2, 1 , tId,tStepCount_,xyzblockSize);
		latticeArray[neibB] = -1*tempAssignNumber;
		auto neibC = getNeighbour(3, 1 , tId,tStepCount_,xyzblockSize);
		latticeArray[neibC] = -1*tempAssignNumber;
	
		printf(" xyzblockSize = %d, posIdx =%d, tID = %d ,xyzblockNumber = %d ,xyzPos = %d [ tIdX %d, tIdY %d, tIdZ %d , bidX:%d, bidY:%d, bidZ:%d ] latticeArray[%d] -> %f ( neibs : %d %d %d %d -> %f %f %f %f ) \n ",\\
			xyzblockSize,posIdx,tId,xyzblockNumber,xyzPos,tIdX,tIdY,tIdZ,blockIdx.x,blockIdx.y,blockIdx.z,posIdx,latticeArray[posIdx],\\
			neib,neibA,neibB,neibC,latticeArray[neib],latticeArray[neibA],latticeArray[neibB],latticeArray[neibC]);
		
		//assert(posIdx < NTot );
		
	 	tId+=2; 	
	}
}

void phiFourLattice::phiFourLatticeGPUConstructor()
{
	cudaMalloc(&CurrentObservablesGPU,latticeSize_);
	cudaMalloc(&CurrentStateGPU,latticeSize_*sizeof(float));
	cout<<" Allocated "<<latticeSize_*sizeof(float)/1024.0<<" Kb of DEVICE Memory for lattice \n";
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
	dim3 blockSize(blockLen_,blockLen_,blockLen_);
	dim3 gridSize(gridLen_,gridLen_,gridLen_);
 
	std::cout<<"Launching the kerrnels for Lattice Size = "<<latticeSize_<<" ( t_d = "<<tStepCount_<<" x_d = "<<xStepCount_<<" & D = "<<dim_<<"\n"
		 <<" with grid size : "<<gridSize.x<<" , "<<gridSize.y<<" , "<<gridSize.z<<"\n"
		 <<" and block size : "<<blockSize.x<<" , "<<blockSize.y<<" , "<<blockSize.z<<"\n";	
	checkBoardUpdate<<<gridSize,blockSize>>>( CurrentStateGPU , 0 , 1.0,tStepCount_ ,latticeSize_ );
	cudaThreadSynchronize();
	cout<<"\n\n_______________________________\n\n";
	//checkBoardUpdate<<<gridSize,blockSize>>>( CurrentStateGPU , 1 , 2.0,tStepCount_ ,latticeSize_ );
	//cudaDeviceSynchronize();
}


