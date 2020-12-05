
#include <math.h>
#include<iostream>
#include<random>
#include<fstream>


#define RANDOM_ENGINE mt19937_64
#define RAND_IDX_MAX 512

#define SWEEP_COUNT 10000

using namespace std;
class lattice
{

	public:
		lattice(double mT=1,double wT=1,int nT=120,double dx=1.0,int randseed=0,int skipSweepCount=-1,int writeEventCount=128);
		~lattice() {} ;
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


