#include "lattice.h"


phiFourLattice::phiFourLattice(uint8_t dim,uint16_t tStepCount,uint16_t xStepCount,
				float mass,float lambda,uint8_t initialization,int randseed) :
						dim_(dim),
						tStepCount_(tStepCount),
						xStepCount_(xStepCount),
						m_(mass),
						lambda_(lambda),
						latticeSize_( dim_> 1 ? tStepCount_*pow(xStepCount_,dim_-1): 0),
						initialization_(initialization),
						randomSeed_(randseed)
{
	std::cout<<"\n dim = "<<dim_<<" ("<< dim <<")"<<" , mass = "<<mass
				<<" , LATTICE SIZE = "<<latticeSize_
				<<" , tStepCount_/xStepCount_ "<<tStepCount_<<"/"<<xStepCount_<<"\n";

	//CurrentStateCPU_= std::make_unique<float>(dim_*tStepCount_*xStepCount_);
	//CurrentObservablesCPU_ = std::make_unique<float>(5);
	//CurrentObservablesCPU= CurrentObservablesCPU_.get();
	//CurrentStateCPU = CurrentStateCPU_.get();
	
	CurrentObservablesCPU= new float[latticeSize_];
	CurrentStateCPU = new float[latticeSize_];

	phiFourLatticeGPUConstructor();
	
	initializeLatticeCPU();

}

phiFourLattice::~phiFourLattice()
{
	delete CurrentObservablesCPU,CurrentObservablesCPU;
	phiFourLatticeGPUDistructor();

}

void phiFourLattice::initializeLatticeCPU(int type,int randseed)
{
	if(initialization_ == 0)
	{
		for(int i=0;i<latticeSize_;i++)
		{
			CurrentObservablesCPU[i]=0.0;
		}
	
	}
	if(type ==1)
	{
		//std::cout<<"\n Doing the hot initialization \n";
		randomSeed_=randseed;
		generator.seed(randomSeed_);

		for(int i=0;i<latticeSize_;i++)
		{
			CurrentStateCPU[i]=dblDistribution(generator);
		//	std::cout<<"  i = "<<i<<"  : "<<CurrentStateCPU[i]<<"\n";
			}
	}
}

void phiFourLattice::initializeLatticeCPU()
{
	if(initialization_ == 0)
	{
		for(int i=0;i<latticeSize_;i++)
		{
			CurrentStateCPU[i]=0.0;
		}
	
	}
}

void phiFourLattice::printLatticeOnCPU()
{
	for(int i=0;i<latticeSize_;i++)
	{
		std::cout<< "  i =  "<<i<<" : " <<CurrentStateCPU[i]<<"\n";
	}
}


void phiFourLattice::writeLatticeToASCII(string fname)
{
	fstream oFile(fname.c_str(),ios::out);

	oFile<<tStepCount_<<","<<xStepCount_<<","<<xStepCount_<<","<<xStepCount_<<"\n";
	for(int i=0;i<tStepCount_;i++)
	for(int j=0;j<xStepCount_;j++)
	for(int k=0;k<xStepCount_;k++)
	for(int l=0;l<xStepCount_;l++)
	{
		auto pos =int(i*pow(xStepCount_,3) +  j*pow(xStepCount_,2) + k*pow(xStepCount_,1) +l );
		oFile<<pos<<"     ,     "<<i<<" , "<<j<<" , "<<k<<" , "<<l<<"      ,     "<<CurrentStateCPU[pos]<<"\n";
	}

	oFile.close();
}
void witeGPUlatticeLayoutToASCII()
{

	const int blockLenX(blockLen_),blockLenY(blockLen_),blockLenZ( blockLen_) ;
	const int gridLenX(gridLen_)  ,gridLenY(gridLen_),  gridLenZ(gridLen_);

	//assert(mode==0 or mode==1);
	int tIdX =  ( threadIdx.x) ;
	int tIdY =  ( threadIdx.y) ;
	int tIdZ =  ( threadIdx.z) ;
	
//	int  xyzPos     = tIdX *gridDim.y*blockDim.y * gridDim.z*blockDim.z + tIdY * gridDim.z*blockDim.z+tIdZ;
	
	for(auto bidX=0;bidX<gridLenX;bidX++)
	for(auto bidX=0;bidX<gridLenX;bidX++)
	for(auto bidX=0;bidX<gridLenX;bidX++)
	{

		for(auto thIdx=0;thidx<blockLenX;thidx++)
		for(auto thIdx=0;thidx<blockLenX;thidx++)
		for(auto thIdx=0;thidx<blockLenX;thidx++)
		{
			
		}

	}
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
		
		printf("xyzblockSize = %d, posIdx =%d, tID = %d ,xyzblockNumber = %d ,xyzPos = %d [ tIdX %d, tIdY %d, tIdZ %d , bidX:%d, bidY:%d, bidZ:%d ] latticeArray[%d] -> %f \n ",
						xyzblockSize,posIdx,tId,xyzblockNumber,xyzPos,tIdX,tIdY,tIdZ,blockIdx.x,blockIdx.y,blockIdx.z,posIdx,latticeArray[posIdx]);
		
		//assert(posIdx < NTot );
		
	 	tId+=2; 	
	}
}



/*
void phiFourLattice::writeLatticeToASCII(string fname)
{
	fstream oFile(fname.c_str(),ios::out);

	oFile<<tStepCount_<<","<<xStepCount_<<","<<xStepCount_<<","<<xStepCount_<<"\n";
	for(int i=0;i<tStepCount_;i++)
	for(int j=0;j<xStepCount_;j++)
	for(int k=0;k<xStepCount_;k++)
	for(int l=0;l<xStepCount_;l++)
	{
		auto pos =int(i*pow(xStepCount_,3) +  j*pow(xStepCount_,2) + k*pow(xStepCount_,1) +l );
		oFile<<pos<<"     ,     "<<i<<" , "<<j<<" , "<<k<<" , "<<l<<"      ,     "<<CurrentStateCPU[pos]<<"\n";
	}

	oFile.close();
}
*/
















































/// Code for the Harmonic Oscilator Problem

 HOLattice::HOLattice(double mT,double wT ,int nT,double dx,int randseed,int skipSweepCount,int writeEventCount)
		: N(nT),mTilda(mT), wTilda(wT),
		  h(dx),idrate(0.8), xVec(N,0.0),
		  skipSweepCount(skipSweepCount>-1 ? skipSweepCount : 0  ),writeOutCount(writeEventCount>-1 ? writeEventCount: 1),
		  alpha(mTilda),beta(mTilda*wTilda*wTilda),
		  initializationSeed(randseed),
		  intDistribution(0,N-1),
		  dblDistribution(-0.5,0.5)
	//	  oFileName(ofname)
{
	initialize();
	oFileName="N"+to_string(N)+"_Nm_"+to_string(int(N*mTilda*1000)/1000)+".txt";

}



void HOLattice::initialize(string type)
{
	generator.seed(initializationSeed);

	if(type=="zero")
	{
		for(int i=0;i<N;i++)
			xVec[i]=0.0;
	}
	if(type=="hot")
	{
		for(int i=0;i<N;i++)
			xVec[i]=dblDistribution(generator);
	}
	populateRandomNumbers();
	findAction();
	writeOutCounter=0;
	skipCounter=0;
	stepCount=0;
	sweepCount=0;
	clearBuff();
	fillBuff();
}

void HOLattice::printLattice(bool printLatticeSites)
{
	cout<<"Printing Lattice with : "<<"\n";
	cout<<"              N  =  "<<N<<"\n";
	cout<<"         mTilda  =  "<<mTilda<<"\n";
	cout<<"         wTilda  =  "<<wTilda<<"\n";
	cout<<"         action  =  "<<action<<"\n";
	cout<<"              h  =  "<<h<<"\n";
	cout<<"         idrate  =  "<<idrate<<"\n";
	cout<<"      stepCount  =  "<<stepCount<<"\n";
	cout<<"     sweepCount  =  "<<sweepCount<<"\n";
	cout<<" skipSweepCount  =  "<<skipSweepCount<<"\n";
	cout<<"  writeOutCount  =  "<<writeOutCount<<"\n";

	if(printLatticeSites)
	{

	cout<<" Lattice is "<<"\n";
	for(int i=0;i<N;i++)
		cout<<" i = " <<i<<" -> "<<xVec[i]<<"\n";
	}
}

void HOLattice::findAction()
{
	double a(0.0),b(0.0);
	for(int i=0;i<N-1;i++)
	{
		a+=(xVec[i+1]-xVec[i])*(xVec[i+1]-xVec[i]);
		b+=xVec[i]*xVec[i];
	}
	
	a+=(xVec[N-1]-xVec[0])*(xVec[N-1]-xVec[0]);
	b+=xVec[N-1]*xVec[N-1];

	action=0.5*mTilda*a+0.5*wTilda*b;
}

void HOLattice::populateRandomNumbers()
{
	
	for(int i=0;i<RAND_IDX_MAX;i++)
	{
		randIdx[i]=intDistribution(generator);
		randVals[i]=dblDistribution(generator);
	}
	randIdCounter=0;
}

void HOLattice::takeSweep(int n)
{
	double percentStep=0.25;
	int oneP=n*percentStep/100;
	double percent =0;
	cout<<"stepCount = "<<stepCount<<" , sweepCount = " <<sweepCount<<" , "<<" skipCounter = "<<skipCounter<<"/"<<skipSweepCount<<" , " ;
	cout<<"writeOutCounter = "<<writeOutCounter<<"/"<<writeOutCount<<"  [ "<<percent<<"%  ]    ";
	percent+=percentStep;	

	for(int k=0;k<n;k++)
	{
	 double accrete=0;
	//	int mm=0;
	
	for(int i=0;i<N;i++)
	{
		auto idx=randIdx[randIdCounter];
		auto deltaX=randVals[randIdCounter]*h;
		randIdCounter++;
		if(randIdCounter==RAND_IDX_MAX)
			populateRandomNumbers();

		auto idxn = idx!=(N-1) ? idx+1 : 0;
		auto idxp = idx!=  0   ? idx-1 : N-1;
		//cout<<idx<<" "<<idxn<<" "<<idxp<<"\n";
		auto deltaS= deltaX*(alpha*(2*xVec[idx]+deltaX-xVec[idxn]-xVec[idxp])+beta*(xVec[idx]+deltaX/2));
		//cout<< deltaS<<" = "<<deltaX<<"*"<<"("<<alpha<<"*"<<"(2*"<<xVec[idx]<<"+"<<deltaX<<"-"<<xVec[idxn]<<"-"<<xVec[idxp]<<")+"<<beta<<"*"<<"("<<xVec[idx]<<"+"<<deltaX<<"/2))";
		
		if(deltaS<0)
		{
			xVec[idx]+=deltaX;
			accrete+= double(1.0/N);
	//		mm+=1;	cout<<"m = "<<mm<<" , i =  "<<i<<" accrete+="<<" double(1.0/"<<N<<") "<< accrete<<" , "<<double(1.0/N)<<"\n";
	
		}
		else if ((dblDistribution(generator)+0.5) < exp(-1*deltaS) )
		{       
			xVec[idx]+=deltaX;
			accrete+= double(1.0/N);
	//		mm+=1;	cout<<"m = "<<mm<<" , i =  "<<i<<" accrete+="<<" double(1.0/"<<N<<") "<< accrete<<" , "<<double(1.0/N)<<"\n";

		}
		else
			deltaS=0;
		//cout<<" dS = "<<deltaS<<" dX = "<<deltaX;
		action+=deltaS;
		//cout<<" action = "<<action<<"\n";
		stepCount+=1;

	}
	
       	        sweepCount+=1;
		skipCounter+=1;
	        
		if(skipCounter>=skipSweepCount)
		{
			fillBuff();
			skipCounter=0;
			writeOutCounter+=1;
			
		}
		if(writeOutCounter==writeOutCount)
		{
			//cout<<"\nwriting out "<<"\n";
			printToASCII(writeOutCount);
			clearBuff();
			writeOutCounter=0;
		}
		
		if(sweepCount%oneP==0)
		{
		cout<<"\r stepCount = "<<stepCount<<" , sweepCount = " <<sweepCount<<" , "<<" skipCounter = "<<skipCounter<<"/"<<skipSweepCount<<" , " ;
	        cout<<"writeOutCounter = "<<writeOutCounter<<"/"<<writeOutCount<<"  [ "<<percent<<"%  ]    ";//<<"\n";
		percent+=percentStep;	
		}
		
		//cout<<" h = "<<h<<" accrete = "<<accrete;//<<" \n";
		h*= accrete/idrate;
	}
	cout<<"\rstepCount = "<<stepCount<<" , sweepCount = " <<sweepCount<<" , "<<" skipCounter = "<<skipCounter<<"/"<<skipSweepCount<<" , " ;
	cout<<"writeOutCounter = "<<writeOutCounter<<"/"<<writeOutCount<<"  [ "<<percent<<"%  ]    ";

}


void HOLattice::printToASCII(int n)
{
	if(n<0)
		n=actiondata.size();

	fstream file(oFileName.c_str(),ios::app);
	file<<"\n#N : "<<N<<"\n";
	file<<"#xVecSize : "<<xVecBuffer.size()<<"\n";
	file<<"#nWrite : "<<n<<"\n";
	auto adataIt =actiondata.begin();
	auto xBuffIt = xVecBuffer.begin();
	auto sweepCountIt =sweepCountData.begin();
	for(long int i=0;i < n; i++)
	{
		//cout<<"\n"<<"printing " << stepCount-(n-i);
		file<<*sweepCountIt<<","<<*adataIt<<",";
		file<<*(xBuffIt);
		for(int j=1;j<N;j++)
		{
			xBuffIt++;
			file<<","<<*(xBuffIt);
		}
		adataIt++;
		sweepCountIt++;
		file<<"\n";

	}

	file.close();
}

void HOLattice::fillBuff()
{
	for(int i=0;i< xVec.size();i++)
		xVecBuffer.push_back(xVec[i]);
	actiondata.push_back(action);
	sweepCountData.push_back(sweepCount);
	stepCountData.push_back(stepCount);
}


void HOLattice::clearBuff()
{
	actiondata.clear();
	xVecBuffer.clear();
	sweepCountData.clear();
	stepCountData.clear();
}
void HOLattice::clearFile()
{
	cout<<"Purging : "<<oFileName<<"\n";
	fstream file(oFileName.c_str(),ios::out);
	file.close();
}

