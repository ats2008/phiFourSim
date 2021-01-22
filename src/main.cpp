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


	// phiFourLattice::phiFourLattice(uint8_t dim,uint16_t tStepCount,uint16_t xStepCount,float a,
	//			float mass,float m2,float lambda,string label ,uint8_t initialization,int randseed,int blockLen) :

	//      phiFourLattice alat(4,16,16,1.0,1.0,"lattice16s8",0,0, 8 );
	//        phiFourLattice alat(4, 14, 14, 1.0 , 2.0 ,-4.0 , 5.113 ,"lattice14b7",0 , 0 , 7 );
                phiFourLattice alat(4, 10, 10, 1.0 , 2.0 ,-4.0 , 6.08  ,"lattice10b2" ,0 , 0 , 2 );
        //      phiFourLattice alat(4,  8,  8, 1.0 , 2.0 ,-4.0 , 6.008 ,"lattice8b4"  ,0 , 0 , 2 );
	//      phiFourLattice alat(4,  4,  4, 0.1 , 2.0 ,-4.0 , 6.008 ,"lattice4b2"  ,0 , 0 , 2 );
	
	int numSteps = (1e5);

	printf("HAHA IN MAIN\n");
	alat.simplePrintfFromKernel();
	

	printf("\nPrint the lattice before heating up \n");
	
	//alat.printLatticeOnGPU();
	
	//alat.initializeLatticeCPU(0,0);
	
	alat.copyStateInCPUtoGPU();
		
	//printf("\n Printing the lattice after the initialization !! \n ");
	//alat.printLatticeOnCPU();
	printf("\n Printing the lattice  on  gpu after the copy !! \n ");
	//alat.printLatticeOnGPU();
	//printf("\n Printing the lattice after the reinitialization and copy !! \n ");
	//alat.printLatticeOnCPU();
	
	alat.doGPUlatticeUpdates(numSteps,true);
	alat.copyStateInGPUtoCPU();
	//alat.printLatticeOnGPU();
	printf("\n Printing the lattice after the GPU update + GPUtoCPU copy !! \n ");
	//alat.printLatticeOnCPU();
	//alat.writeLatticeToASCII("alattice.txt");
	alat.writeGPUlatticeLayoutToASCII("alattice.txt");

	return 0;
}
