#include "lattice.h"
#include "iostream"

using namespace std;

int main(int argc,char *argv[])
{

	double m=1,mN=120;
	int randseed=0;
	int skipSweepCount =100;
	int writeEventCount=128;
	double h=1;
	long int eventsRequired=8;
	long int sweepMaxCount(eventsRequired*skipSweepCount);

	if(argc>1)
		m=atof(argv[1]);
	if(argc>2)
		mN=atof(argv[2]);
	if(argc>3)
		skipSweepCount=int(atof(argv[3]));
	if(argc>4)
		eventsRequired=atof(argv[4]);


	//lattice(double mT=1,double wT=1,int nT=120,double dx=1.0,int randseed=0,int skipSweepCount=-1,int writeEventCount=128);
//	phiFourLattice(uint8_t dim=4,uint16_t tStepCount_=8,uint16_t xStepCount=8,
//				float mass=1,float lambda=1,uint8_t initialization=0,int randseed=0) ;
	phiFourLattice alat(4,4,4);
	
	printf("HAHA IN MAIN\n");
	//alat.simplePrintfFromKernel();

	printf("\nPrint the lattice before heating up \n");
	
	//alat.printLatticeOnGPU();
	
	alat.initializeLatticeCPU(0,0);
	
	alat.copyStateInCPUtoGPU();
		
	printf("\n Printing the lattice after the initialization !! \n ");
	alat.printLatticeOnCPU();
	//printf("\n Printing the lattice  on  gpu after the copy !! \n ");
	//alat.printLatticeOnGPU();
	printf("\n Printing the lattice after the reinitialization and copy !! \n ");
	//alat.printLatticeOnCPU();
	
	alat.doGPUlatticeUpdates(2);
	alat.copyStateInGPUtoCPU();
	alat.printLatticeOnGPU();
	printf("\n Printing the lattice after the GPU update + GPUtoCPU copy !! \n ");
	alat.printLatticeOnCPU();
	alat.writeLatticeToASCII("alattice.txt");

	return 0;
}
