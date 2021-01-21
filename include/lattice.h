#include <math.h>
#include<iostream>
#include<random>
#include<fstream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


#define RANDOM_ENGINE mt19937_64
#define RAND_IDX_MAX 512

#define MAX_RGEN_STEP 1024

#define SWEEP_COUNT 10000

using namespace std;


class Lattice
{


};



class phiFourLattice : public Lattice {

	public :
		
		constexpr static uint8_t DIMMAX =4;
		
		phiFourLattice(uint8_t dim=4,uint16_t tStepCount_=8,uint16_t xStepCount=8,
				float mass=1,float lambda=1,string label="blattice", uint8_t initialization=0,int randseed=0,int blockLen_=8) ;
		~phiFourLattice();

		void simplePrintfFromKernel();
		
		void copyStateInCPUtoGPU();
		void copyStateInGPUtoCPU();
		void copyObservalblesInGPUToaCPU();
		void copyObservalblesInCPUToGPU();
		void copyBufferToCPU(int begi=0,int end=1);
		void printLatticeOnGPU();
		void printLatticeOnCPU();
		void writeLatticeToASCII(string fname="LAT.txt");
		void writeGPUlatticeLayoutToASCII(string fname="LAT.txt");
		void writeBufferToFileGPULayout(string fname="LATbuff.txt",int beg=0,int end=1 );	

		void initializeLatticeCPU(int type,int randseed);
		void doGPUlatticeUpdates(int numUpdates=1,bool copyToCPU=false);

	private :
		
		const string latticeLabel;
		const int dim_;
		const uint16_t tStepCount_;
		const uint16_t xStepCount_;
		const float m_,lambda_;
		const uint32_t latticeSize_;	
		const uint8_t initialization_;
		const int writeFileMax;
		      int writeFileCount;
		const int blockLen_;
		int gridLen_;
		
		uint32_t randomSeed_;
		RANDOM_ENGINE generator;	
		uniform_int_distribution<int> intDistribution;	
		uniform_real_distribution<double> dblDistribution;	
	
		float * gpuUniforRealRandomBank;
		curandState *RNG_State;
		const int maxStepCountForSingleRandomNumberFill;
		int currentStep;
		void fillGPURandomNumberBank();

		const int bufferSize,obsevablesCount;
		int currentBufferPosGPU,currentBufferPosCPU;
		std::unique_ptr<float> CurrentStateCPU_;
		std::unique_ptr<float> CurrentObservablesCPU_;
		float *CurrentStateCPU,*CurrentObservablesCPU;
		float *CurrentStateGPU,*CurrentObservablesGPU;
		float *StatesBufferCPU,*ObservablesBufferCPU,*EnergyBufferCPU;
		float *StatesBufferGPU,*ObservablesBufferGPU;
		
		float *gpuDeltaEworkspace;

		void initializeLatticeCPU();
		void initializeLatticeGPU();

		void phiFourLatticeGPUConstructor();
		void phiFourLatticeGPUDistructor();

		enum obs{S};
	
		
};
	



class HOLattice
{

	public:
		HOLattice(double mT=1,double wT=1,int nT=120,double dx=1.0,int randseed=0,int skipSweepCount=-1,int writeEventCount=128);
		~HOLattice() {} ;
		void initialize(string type="zero");
		void takeSweep(int nSteps);

		void printToASCII(int n=-1);
		void printLattice(bool printLatticeSites= false);
		void clearBuff();
		void clearFile();

	private:
		int N;
		double mTilda,wTilda;
		vector<double> xVec;
	        double action;
		double h,idrate;
		int initializationSeed;
		int skipSweepCount;
		int skipCounter;
		int writeOutCounter;
		int writeOutCount;
		long int stepCount;
		long int sweepCount;
		vector<double> actiondata;
		vector<double> xVecBuffer;
		vector<long int> stepCountData;
		vector<long int> sweepCountData;
		double alpha,beta;
	
		void findAction();
		void fillBuff();
		RANDOM_ENGINE generator;	
		uniform_int_distribution<int> intDistribution;	
		uniform_real_distribution<double> dblDistribution;	
		int randIdCounter;
		int randIdx[RAND_IDX_MAX];
		double randVals[RAND_IDX_MAX];
		void populateRandomNumbers();
		

		string oFileName;
};
	
		
	






