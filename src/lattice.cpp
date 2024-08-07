#include "lattice.h"


phiFourLattice::phiFourLattice(uint8_t dim,uint16_t tStepCount,uint16_t xStepCount,float a,
				float mass,float m2,float lambda,string label ,uint8_t initialization,int randseed,int blockLen, int thermalizeSkip, int acfSkip) :
						latticeLabel(label),
						dim_(dim),
						tStepCount_(tStepCount),
						xStepCount_(xStepCount),
						a_(a),
						m_(mass),
						m2_(m2),
						lambda_(lambda),
						latticeSize_( dim_> 1 ? tStepCount_*pow(xStepCount_,dim_-1): 0),
						initialization_(initialization),
						randomSeed_(randseed),
						blockLen_(blockLen),
						maxStepCountForSingleRandomNumberFill(MAX_RGEN_STEP),
						currentStep(0),
						bufferSize(1024),
						obsevablesCount(5),
						currentBufferPosGPU(0),
						currentBufferPosCPU(0),
						writeFileMax(256),
						writeFileCount(0),
						mTilda_(m_*a),
						m2Tilda_(m2_*a*a),
						lTilda_(lambda*a*a*a*a),
						thermalizationSkip_(thermalizeSkip),
						autoCorrSkip_(acfSkip)
{
	std::cout<<"\n dim = "<<dim_<<" ("<< dim <<")"<<" , mass = "<<mass
				<<" , LATTICE SIZE = "<<latticeSize_
				<<" , tStepCount_/xStepCount_ "<<tStepCount_<<"/"<<xStepCount_<<"\n";

	//CurrentStateCPU_= std::make_unique<float>(dim_*tStepCount_*xStepCount_);
	//CurrentObservablesCPU_ = std::make_unique<float>(5);
	//CurrentObservablesCPU= CurrentObservablesCPU_.get();
	//CurrentStateCPU = CurrentStateCPU_.get();
	
	StatesBufferCPU       = new float[latticeSize_*bufferSize];
	ObservablesBufferCPU  = new float[obsevablesCount * bufferSize];
	EnergyBufferCPU        = new float[bufferSize];
	CurrentStateCPU       = StatesBufferCPU;
	CurrentObservablesCPU = ObservablesBufferCPU;


	gridLen_=( xStepCount_/blockLen );
	cout<<" Allocated "<<bufferSize*latticeSize_*sizeof(float)/1024.0/1024<<" MB of HOST Memory for lattice \n";
	cout<<" Allocated "<<bufferSize*obsevablesCount*sizeof(float)/1024.0/1024<<" MB of HOST Memory for observable buffer \n";
	cout<<" Allocated "<<bufferSize*sizeof(float)/1024.0/1024<<" MB of HOST Memory for Energy buffer \n";

	phiFourLatticeGPUConstructor();
	
	initializeLatticeCPU();

}

phiFourLattice::~phiFourLattice()
{
	delete[] StatesBufferCPU;
	delete[] ObservablesBufferCPU;
	delete[] EnergyBufferCPU;
	phiFourLatticeGPUDistructor();

}

void phiFourLattice::initializeLatticeCPU(int type,int randseed)
{
	if(type == 0)
	{
		std::cout<<"Doing the T=0 initialization \n";
		for(int i=0;i<latticeSize_;i++)
		{
			CurrentStateCPU[i]=0.0;
		}
	
	}
	if(type ==1)
	{
		std::cout<<"\n Doing the hot initialization \n";
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
		std::cout<<"Doing the T=0 initialization \n";
		for(int i=0;i<latticeSize_;i++)
		{
			CurrentStateCPU[i]=0.0;
		}
	
	}

	if(initialization_ ==1)
	{
		std::cout<<"\n Doing the hot initialization \n";
		generator.seed(randomSeed_);

		for(int i=0;i<latticeSize_;i++)
		{
			CurrentStateCPU[i]=dblDistribution(generator);
		//	std::cout<<"  i = "<<i<<"  : "<<CurrentStateCPU[i]<<"\n";
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

void phiFourLattice::writeGPUlatticeLayoutToASCII(string fname)
{
	fstream oFile(fname.c_str(),ios::out);

	const int blockLenX(blockLen_),blockLenY(blockLen_),blockLenZ( blockLen_) ;
	const int gridLenX(gridLen_)  ,gridLenY(gridLen_),  gridLenZ(gridLen_);
	const int xyzblockSize = blockLenX*blockLenY*blockLenZ;
	
	oFile<<tStepCount_<<","<<xStepCount_<<","<<xStepCount_<<","<<xStepCount_<<"\n";
	
	for(auto bidX=0;bidX<gridLenX;bidX++)
	for(auto bidY=0;bidY<gridLenY;bidY++)
	for(auto bidZ=0;bidZ<gridLenZ;bidZ++)
	{

		for(auto thIdx=0;thIdx<blockLenX;thIdx++)
		for(auto thIdy=0;thIdy<blockLenY;thIdy++)
		for(auto thIdz=0;thIdz<blockLenZ;thIdz++)
		{
			auto xyzblockNumber = bidX*gridLenY*gridLenZ + bidY*gridLenZ + bidZ;
			auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
						+ thIdx*blockLenY*blockLenZ + thIdy*blockLenZ +thIdz ;
		
			auto x = bidX*blockLenX + thIdx ;
			auto y = bidY*blockLenY + thIdy ;
			auto z = bidZ*blockLenZ + thIdz ;

			for(auto t=0;t<tStepCount_;t++)
			{
				auto pos = t*xyzblockSize + xyzPos ;
				oFile<<pos<<"     ,     "<<t<<" , "<<x<<" , "<<y<<" , "<<z<<"      ,     "<<CurrentStateCPU[pos]<<"\n";
			}
		}

	}
	
	oFile.close();
}


void phiFourLattice::writeBufferToFileGPULayout(string fname,int beg,int end,bool writeLattice)
{
	const int blockLenX(blockLen_),blockLenY(blockLen_),blockLenZ( blockLen_) ;
	const int gridLenX(gridLen_)  ,gridLenY(gridLen_),  gridLenZ(gridLen_);
	const int xyzblockSize = blockLenX*blockLenY*blockLenZ;
			
	writeFileCount++;
	fname=latticeLabel;
	string prifix("data/");
	fstream	oFile((prifix+fname+"_"+to_string(writeFileCount)+".txt").c_str(),ios::out);
	fstream oObsFile((prifix+"obs_"+fname+"_"+to_string(writeFileCount)+".txt").c_str(),ios::out);
	if(writeLattice)
		oFile<<tStepCount_<<","<<xStepCount_<<","<<xStepCount_<<","<<xStepCount_<<"\n";
	
	oObsFile<<tStepCount_<<","<<xStepCount_<<","<<xStepCount_<<","<<xStepCount_<<"\n";
	oObsFile<<obsevablesCount<<"\n";
	
	cout<<"Writing out buffer from "<<beg<<"  to  "<<end<<" to "<<fname+"_"+to_string(writeFileCount)+".txt"<<"\n";
	int wcount=0;
	for(int i=0;i<(end-beg);i++)
	{
		if(wcount== writeFileMax)
		{
			writeFileCount++;
			wcount=0;
	
			oFile.close();
			oFile.open(prifix+(fname+"_"+to_string(writeFileCount)+".txt").c_str(),ios::out);
			oFile<<tStepCount_<<","<<xStepCount_<<","<<xStepCount_<<","<<xStepCount_<<"\n";
			
			oObsFile.close();
			oObsFile.open((prifix+"obs_"+fname+"_"+to_string(writeFileCount)+".txt").c_str(),ios::out);
			oObsFile<<tStepCount_<<","<<xStepCount_<<","<<xStepCount_<<","<<xStepCount_<<"\n";
			oObsFile<<obsevablesCount<<"\n";
			}
		wcount++;
		auto offset = latticeSize_*i;
		oFile<<"!"<<i<<"\n";
		for(auto bidX=0;bidX<gridLenX;bidX++)
		for(auto bidY=0;bidY<gridLenY;bidY++)
		for(auto bidZ=0;bidZ<gridLenZ;bidZ++)
		{
			if(writeLattice)
			for(auto thIdx=0;thIdx<blockLenX;thIdx++)
			for(auto thIdy=0;thIdy<blockLenY;thIdy++)
			for(auto thIdz=0;thIdz<blockLenZ;thIdz++)
			{
				auto xyzblockNumber = bidX*gridLenY*gridLenZ + bidY*gridLenZ + bidZ;
				auto xyzPos         = ( xyzblockSize * xyzblockNumber * tStepCount_ ) 
							+ thIdx*blockLenY*blockLenZ + thIdy*blockLenZ +thIdz ;
			
				auto x = bidX*blockLenX + thIdx ;
				auto y = bidY*blockLenY + thIdy ;
				auto z = bidZ*blockLenZ + thIdz ;

				for(auto t=0;t<tStepCount_;t++)
				{
					auto pos = t*xyzblockSize + xyzPos ;
					oFile<<pos<<"     ,     "<<t<<" , "<<x<<" , "<<y<<" , "<<z<<"      ,     "<<StatesBufferCPU[pos+offset]<<"\n";
				}
			}

		}
		oObsFile<<i<<","<<EnergyBufferCPU[i];
		offset = obsevablesCount*i;
		for(int j=0;j<obsevablesCount;j++)
		{
			oObsFile<<","<<ObservablesBufferCPU[j+offset];
		}
		oObsFile<<"\n";
	}

	oFile.close();
	oObsFile.close();
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

