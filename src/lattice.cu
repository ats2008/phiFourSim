#include "lattice.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

__global__ void testrun()
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	printf("global tId : %d \n",tId);
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
		auto neibBlkIdx = ( threadIdx.x + dir < 0 or threadIdx.x + dir > blockDim.x -1 )? (blockIdx.x + dir + gridDim.x )%gridDim.x : blockIdx.x;
		auto neibThrIdx = ( threadIdx.x + dir + blockDim.x )%blockDim.x;
		
		auto xyzblockNumber = neibBlkIdx*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
		auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ neibThrIdx*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +threadIdx.z ;
		auto posIdx= tId*xyzblockSize + xyzPos ;
		
		return posIdx;
	}
	if(probeDim == 2 ) 
	{
		auto neibBlkIdx = ( threadIdx.y +dir < 0 or threadIdx.y + dir > blockDim.y -1 )? (blockIdx.y + dir + gridDim.y )%gridDim.y : blockIdx.y;
		auto neibThrIdx = ( threadIdx.y+ dir + blockDim.y )%blockDim.y;
		
		auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + neibBlkIdx*gridDim.z + blockIdx.z;
		auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + neibThrIdx*blockDim.z +threadIdx.z ;
		auto posIdx= tId*xyzblockSize + xyzPos ;
		
		return posIdx;
	}
	if(probeDim == 3 ) 
	{
		auto neibBlkIdx = (threadIdx.z +dir < 0 or threadIdx.z + dir > blockDim.z -1 )? (blockIdx.z + dir + gridDim.z )%gridDim.z : blockIdx.z;
		auto neibThrIdx = (threadIdx.z + dir + blockDim.z )%blockDim.z;
		
		auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + neibBlkIdx;
		auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +neibThrIdx ;
		auto posIdx= tId*xyzblockSize + xyzPos ;
		
		return posIdx;
	}
	
	return -1;
}

__global__ void checkBoardPhiFourUpdate(float* neiblatticeArray,float* currentlatticeArray,float* destlatticeArray,float *deltaEworkspace,\\
					float m2,float lambda,int mode,float tempAssignNumber, int tStepCount_, const int NTot,float * RNG_bank, float RWidth =2.0)
{
	
	auto xyzblockSize   = blockDim.x*blockDim.y*blockDim.z;
	auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
	auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +threadIdx.z ;
		
	auto tId = (threadIdx.x + threadIdx.y + threadIdx.z ) % 2 ;
	
	if( mode==1 && tId==0 ) tId=1;
	else if( mode==1 && tId==1 ) tId=0;
	
	while(tId<tStepCount_)
	{
		
		auto posIdx= tId*xyzblockSize + xyzPos ;
		auto phix=currentlatticeArray[posIdx];
		
		float dPhi= RWidth*(RNG_bank[2*posIdx]-0.5);
		auto neibPlus =( (tId+1 + tStepCount_)%tStepCount_)*xyzblockSize + xyzPos  ;
		auto neibMinus=( (tId-1 + tStepCount_)%tStepCount_)*xyzblockSize + xyzPos  ;
		
		float deltaE=2*(2*phix + dPhi )*dPhi- dPhi*(neiblatticeArray[neibPlus] + neiblatticeArray[neibMinus]);
		
		neibPlus  = getNeighbour(1,  1 , tId,tStepCount_,xyzblockSize);
		neibMinus = getNeighbour(1, -1 , tId,tStepCount_,xyzblockSize);
		deltaE   += 2*(2*phix + dPhi )*dPhi- dPhi*(neiblatticeArray[neibPlus] + neiblatticeArray[neibMinus]);
		
		neibPlus  = getNeighbour(2,  1 , tId,tStepCount_,xyzblockSize);
		neibMinus = getNeighbour(2, -1 , tId,tStepCount_,xyzblockSize);
		deltaE   += 2*(2*phix + dPhi )*dPhi- dPhi*(neiblatticeArray[neibPlus] + neiblatticeArray[neibMinus]);
		
		neibPlus  = getNeighbour(3,  1 , tId,tStepCount_,xyzblockSize);
		neibMinus = getNeighbour(3, -1 , tId,tStepCount_,xyzblockSize);
		deltaE   += 2*(2*phix + dPhi )*dPhi- dPhi*(neiblatticeArray[neibPlus] + neiblatticeArray[neibMinus]);
		
		deltaE   += ( (phix+dPhi)*(phix+dPhi) -phix*phix )*m2 + lambda*((phix+dPhi)*(phix+dPhi)*(phix+dPhi)*(phix+dPhi) -phix*phix*phix*phix );

		if(deltaE<0 || ( deltaE>0 && (exp(-deltaE) > RNG_bank[2*posIdx+1])) )
		{
			destlatticeArray[posIdx]  = phix+dPhi;
			deltaEworkspace[posIdx]   = deltaE;
		//	deltaEworkspace[posIdx]   = dPhi;
		}
		else
		{
			destlatticeArray[posIdx]  = phix;
			deltaEworkspace[posIdx]   = 0.0;
		
		}

		//printf(" posIdx : %d , dE : %f ,dPhi : %f, phiOld : %f , phiFinal : %f , exp(-dE) : %f , rnd : %f \n",\\
				posIdx,deltaE,dPhi,currentlatticeArray[posIdx],destlatticeArray[posIdx] ,exp(-deltaE),RNG_bank[posIdx+1]);
	 	tId+=2; 	
	}
}


__global__ void checkBoardUpdate(float* latticeArray,int mode,float tempAssignNumber, int tStepCount_, const int NTot,float * RNG_bank)
{
	
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
		latticeArray[posIdx]=RNG_bank[posIdx];
		
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
			xyzblockSize,posIdx,tId,xyzblockNumber,xyzPos,threadIdx.x,threadIdx.y,threadIdx.z,\\
			blockIdx.x,blockIdx.y,blockIdx.z,posIdx,latticeArray[posIdx],\\
			neib,neibA,neibB,neibC,latticeArray[neib],latticeArray[neibA],latticeArray[neibB],latticeArray[neibC]);
		
		//assert(posIdx < NTot );
		
	 	tId+=2; 	
	}
}

__global__ void init_RNG(curandState* RNG_State,int tStepCount_,int arraySize)
{
	auto xyzblockSize   = blockDim.x*blockDim.y*blockDim.z;
	auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
	auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +threadIdx.z ;

	for(int tId=0;tId<tStepCount_;tId++)
	{
		auto posIdx= tId*xyzblockSize + xyzPos ;
		if(posIdx< arraySize)
			curand_init(1337,posIdx,0,RNG_State+posIdx);

	}
}

__global__ void make_rand(curandState* RNG_State ,float*randArray,int tStepCount_,int buffSize,int latticeSize_ ,int arraySize)
{
	auto xyzblockSize   = blockDim.x*blockDim.y*blockDim.z;
	auto xyzblockNumber = blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z + blockIdx.z;
	auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
				+ threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z +threadIdx.z ;
	for(int buffId=0;buffId<buffSize;buffId++)
	for(int tId=0;tId<tStepCount_;tId++)
	{
		auto xyztPosIdx = (tId*xyzblockSize + xyzPos);
		auto posIdx= buffId*latticeSize_+ xyztPosIdx ;
		if(posIdx< arraySize)
		{
			randArray[2*posIdx]=curand_uniform(&RNG_State[xyztPosIdx]);
			randArray[2*posIdx+1]=curand_uniform(&RNG_State[xyztPosIdx]);
		}
	}
}

void phiFourLattice::phiFourLatticeGPUConstructor()
{
	auto err = cudaMalloc(&ObservablesBufferGPU,bufferSize*obsevablesCount*sizeof(float));
	cudaDeviceSynchronize();
	if(err)
		cout<<cudaGetErrorName(err)<<" : "<<cudaGetErrorString(err)<<"\n";
	
	err=cudaMalloc(&StatesBufferGPU,bufferSize*latticeSize_*sizeof(float));
	if(err) 	cout<<cudaGetErrorName(err)<<" : "<<cudaGetErrorString(err)<<"\n";
	
	err=cudaMalloc(&gpuDeltaEworkspace,latticeSize_*sizeof(float));
	if(err) 	cout<<cudaGetErrorName(err)<<" : "<<cudaGetErrorString(err)<<"\n";


	CurrentStateGPU = StatesBufferGPU;
	CurrentObservablesGPU = ObservablesBufferGPU;
	
	dim3 blockSize(blockLen_,blockLen_,blockLen_);
	dim3 gridSize(gridLen_,gridLen_,gridLen_);

	auto RNG_bankSize = maxStepCountForSingleRandomNumberFill*latticeSize_*2;
	err=cudaMalloc(&RNG_State,latticeSize_*sizeof(curandState));
	if(err)		cout<<cudaGetErrorName(err)<<" : "<<cudaGetErrorString(err)<<"\n";

	err=cudaMalloc(&gpuUniforRealRandomBank,RNG_bankSize*sizeof(float));
	if(err)		cout<<cudaGetErrorName(err)<<"@ cuMalloc gpuUniforRealRandomBank : "<<cudaGetErrorString(err)<<"\n";
	
	cout<<" Initializing random states with "<<latticeSize_<<" seeds ";
	init_RNG<<<gridSize , blockSize >>>(RNG_State,tStepCount_,latticeSize_);
	err=cudaGetLastError();
	if(err) cout<<cudaGetErrorName(err)<<"@ init_RNG : "<<cudaGetErrorString(err)<<"\n";
	
	fillGPURandomNumberBank();
	
	auto memsize=bufferSize*latticeSize_*sizeof(float);
	cout<<" Allocated "<<bufferSize*latticeSize_*sizeof(float)/1024.0/1024.0<<" MB of DEVICE Memory for lattice ( buffer size :  "<<bufferSize<<" ) \n";
	memsize+=bufferSize*obsevablesCount*sizeof(float);
	cout<<" Allocated "<<bufferSize*obsevablesCount*sizeof(float)/1024.0/1024.0<<" MB of DEVICE Memory for  obsevables ( buffer size :  "<<bufferSize<<" @ "<<obsevablesCount<<" ) \n";
	memsize+=latticeSize_*sizeof(float);
	cout<<" Allocated "<<latticeSize_*sizeof(float)/1024.0/1024.0<<" MB of DEVICE Memory for  gpuDeltaEworkspace \n";
	memsize+=latticeSize_*sizeof(curandState);
	cout<<" Allocated "<<latticeSize_*sizeof(curandState)/1024.0/1024.0<<" MB of DEVICE Memory for  RNG_State \n";
	memsize+=RNG_bankSize*sizeof(float);
	cout<<" Allocated "<<RNG_bankSize*sizeof(float)/1024.0/1024.0<<" MB of DEVICE Memory for  gpuUniforRealRandomBank \n";

	cout<<" Total Allocated Device Memory = "<<memsize/1024.0/1024.0  <<" MB \n";
}

void phiFourLattice::fillGPURandomNumberBank()
{
	auto RNG_bankSize = maxStepCountForSingleRandomNumberFill*latticeSize_;
	dim3 blockSize(blockLen_,blockLen_,blockLen_);
	dim3 gridSize(gridLen_,gridLen_,gridLen_);
	cout<<"Filling gpuUniforRealRandomBank with "<<RNG_bankSize<<"new random numbers\n";
	auto RNG_bufferStrides=maxStepCountForSingleRandomNumberFill;
	make_rand<<<gridSize , blockSize >>>(RNG_State,gpuUniforRealRandomBank,tStepCount_,RNG_bufferStrides,latticeSize_,RNG_bankSize);
	auto err=cudaGetLastError();
	if(err)		cout<<cudaGetErrorName(err)<<"@ make_rand : "<<cudaGetErrorString(err)<<"\n";
	cudaDeviceSynchronize();

}

void phiFourLattice::phiFourLatticeGPUDistructor()
{
	cudaFree(gpuUniforRealRandomBank);
	cudaFree(RNG_State);
	cudaFree(StatesBufferGPU);
	cudaFree(ObservablesBufferGPU);
	cudaFree(gpuDeltaEworkspace);
}

void phiFourLattice::simplePrintfFromKernel()
{

	auto err=cudaGetLastError();
	printf("checking the err before launch \n");
	if(err)
		cout<<cudaGetErrorName(err)<<" before testrun : "<<cudaGetErrorString(err)<<"\n";
	printf("haha in the wraper \n");
	testrun<<<5,2>>>();
	cudaDeviceSynchronize();
	err=cudaGetLastError();
	cout<<cudaGetErrorName(err)<<" after testrun : "<<cudaGetErrorString(err)<<"\n";
	printf("haha leaving the wraper \n");

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

void phiFourLattice::copyBufferToCPU(int begi,int end)
{
	cudaMemcpy(StatesBufferCPU,&StatesBufferGPU[begi],(end-begi)*latticeSize_*sizeof(float),cudaMemcpyDeviceToHost); 	
	cudaMemcpy(ObservablesBufferCPU,&ObservablesBufferGPU[begi],(end-begi)*obsevablesCount*sizeof(float),cudaMemcpyDeviceToHost); 	

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

void phiFourLattice::doGPUlatticeUpdates( int numUpdates,bool copyToCPU)
{
	dim3 blockSize(blockLen_,blockLen_,blockLen_);
	dim3 gridSize(gridLen_,gridLen_,gridLen_);
 
	std::cout<<"Launching the kerrnels for Lattice Size = "<<latticeSize_<<" ( t_d = "<<tStepCount_<<" x_d = "<<xStepCount_<<" & D = "<<dim_<<"\n"
		 <<" with grid size : "<<gridSize.x<<" , "<<gridSize.y<<" , "<<gridSize.z<<"\n"
		 <<" and block size : "<<blockSize.x<<" , "<<blockSize.y<<" , "<<blockSize.z<<"\n";	
	
	//thrust::device_ptr<float> cptr = thrust::device_pointer_cast(gpuDeltaEworkspace);

	thrust::device_ptr<float> thrust_ptr_ToDeltaEWplaceB = thrust::device_pointer_cast(gpuDeltaEworkspace);
	float currEnergy = thrust::reduce(thrust_ptr_ToDeltaEWplaceB,thrust_ptr_ToDeltaEWplaceB + latticeSize_  );
	
	EnergyBufferCPU[currentBufferPosCPU]= currEnergy/latticeSize_;
	
	for(int i=0;i<numUpdates;i++)
	{
	
		CurrentStateGPU 	= StatesBufferGPU + latticeSize_*currentBufferPosGPU;
		CurrentObservablesGPU	= ObservablesBufferGPU + obsevablesCount*currentBufferPosGPU;
		currentBufferPosGPU++;
		currentBufferPosCPU++;
		if(currentBufferPosCPU == bufferSize) 	
		{
			currentBufferPosCPU=0;
		}

		if(currentBufferPosGPU == bufferSize) 	
		{
			
			if( copyToCPU )
			{
				cudaDeviceSynchronize();
				copyBufferToCPU(0,currentBufferPosGPU);
				cudaDeviceSynchronize();
				writeBufferToFileGPULayout("blattice",0,currentBufferPosGPU);
			}
			currentBufferPosGPU=0;
		}
		
		auto neibLattice   = CurrentStateGPU;
		auto phixLattice   = CurrentStateGPU;
		auto destnLattice  = StatesBufferGPU + currentBufferPosGPU*latticeSize_;
		checkBoardPhiFourUpdate<<<gridSize,blockSize>>>( neibLattice , phixLattice , destnLattice, gpuDeltaEworkspace , \\
					 	m2Tilda_, lTilda_, 0 , 1.0,tStepCount_ ,latticeSize_, &gpuUniforRealRandomBank[2*currentStep*latticeSize_]);
		neibLattice   = destnLattice;
		phixLattice   = CurrentStateGPU;
		destnLattice  = destnLattice;

		checkBoardPhiFourUpdate<<<gridSize,blockSize>>>(  neibLattice , phixLattice , destnLattice, gpuDeltaEworkspace , \\
						m2Tilda_, lTilda_, 1 , 1.0,tStepCount_ ,latticeSize_, &gpuUniforRealRandomBank[2*currentStep*latticeSize_]);
		currEnergy += thrust::reduce(thrust_ptr_ToDeltaEWplaceB,thrust_ptr_ToDeltaEWplaceB + latticeSize_  );
		cudaDeviceSynchronize();
		EnergyBufferCPU[currentBufferPosCPU] = currEnergy/latticeSize_;
		if(i%256==0)
		{	
			std::cout<<"Doing "<<i<<" th iter \n";
			cout<<"E = "<<currEnergy<< " , currentBufferPosCPU = "<<currentBufferPosCPU<<"  EnergyBufferCPU[currentBufferPosCPU]  : "<<EnergyBufferCPU[currentBufferPosCPU]<<"\n";
		}

		currentStep++;
		if(currentStep==maxStepCountForSingleRandomNumberFill)
		{
			fillGPURandomNumberBank();
			currentStep=0;
		}

	}

	if( copyToCPU )
	{
		cudaDeviceSynchronize();
		copyBufferToCPU(0,currentBufferPosGPU);
		writeBufferToFileGPULayout("blattice",0,currentBufferPosGPU);
	}

	cout<<"\n\n_______________________________\n\n";
}

/*
void phiFourLattice::doGPUlatticeUpdates( int numUpdates)
{
	dim3 blockSize(blockLen_,blockLen_,blockLen_);
	dim3 gridSize(gridLen_,gridLen_,gridLen_);
 
	std::cout<<"Launching the kerrnels for Lattice Size = "<<latticeSize_<<" ( t_d = "<<tStepCount_<<" x_d = "<<xStepCount_<<" & D = "<<dim_<<"\n"
		 <<" with grid size : "<<gridSize.x<<" , "<<gridSize.y<<" , "<<gridSize.z<<"\n"
		 <<" and block size : "<<blockSize.x<<" , "<<blockSize.y<<" , "<<blockSize.z<<"\n";	
	
	//checkBoardUpdate<<<gridSize,blockSize>>>( CurrentStateGPU , 0 , 1.0,tStepCount_ ,latticeSize_, gpuUniforRealRandomBank );
	
	checkBoardPhiFourUpdate<<<gridSize,blockSize>>>( CurrentStateGPU , 0 , 1.0,tStepCount_ ,latticeSize_, &gpuUniforRealRandomBank[0]);
	cudaDeviceSynchronize();
	cout<<"\n\n_______________________________\n\n";
	//checkBoardUpdate<<<gridSize,blockSize>>>( CurrentStateGPU , 1 , 2.0,tStepCount_ ,latticeSize_ );
	//cudaDeviceSynchronize();
}
*/

