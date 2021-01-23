#include "lattice.h"
#include "iostream"

using namespace std;

int main(int argc,char *argv[])
{

	int randseed=0;
	int skipSweepCount =100;
	int writeEventCount=128;
	double h=1;
	long int eventsRequired=8;
	long int sweepMaxCount(eventsRequired*skipSweepCount);
	float choice=5.0;
	if(argc>1)
		choice=atof(argv[1]);
	/*if(argc>2)
		mN=atof(argv[2]);
	if(argc>3)
		skipSweepCount=int(atof(argv[3]));
	if(argc>4)
		eventsRequired=atof(argv[4]);
	*/

	// phiFourLattice::phiFourLattice(uint8_t dim,uint16_t tStepCount,uint16_t xStepCount,float a,
	//			float mass,float m2,float lambda,string label ,uint8_t initialization,int randseed,int blockLen) :
	
	 int dim(4);
	 int tStep(4);
	 int xStep(4);
	 float a(1.0);
	 float m(2.0);
	 float m2(-4.0);
	 float lambda(6.008);
	 string label("lattice4b2");
	 int init(0);
	 int rSeed(0);
	 int blockLen(2);

	 if(choice== 1.0)      {dim=  4;tStep=  16;xStep=16;a= 1.0 ;m= 1.0 ; m2=-4.0 ;lambda= 5.034 ;label="lattice16s2" ;init=0; rSeed=  0 ;blockLen= 2 ;}
	 if(choice== 2.0)      {dim=  4;tStep=  14;xStep=14;a= 1.0 ;m= 2.0 ; m2=-4.0 ;lambda= 5.113 ;label="lattice14b2" ;init=0; rSeed=  0 ;blockLen= 2 ;}
         if(choice== 3.0)      {dim=  4;tStep=  10;xStep=10;a= 1.0 ;m= 2.0 ; m2=-4.0 ;lambda= 6.08  ;label="lattice10b2" ;init=0; rSeed=  0 ;blockLen= 2 ;}
         if(choice== 4.0)      {dim=  4;tStep=   8;xStep= 8;a= 1.0 ;m= 2.0 ; m2=-4.0 ;lambda= 6.008 ;label="lattice8b2"  ;init=0; rSeed=  0 ;blockLen= 2 ;}
	 if(choice== 5.0)      {dim=  4;tStep=   4;xStep= 4;a= 1.0 ;m= 2.0 ; m2=-4.0 ;lambda= 6.008 ;label="lattice4b2"  ;init=0; rSeed=  0 ;blockLen= 2 ;}
	
	int numSteps = (1000);
 	
	phiFourLattice alat( dim , tStep, xStep , a  , m , m2 , lambda ,label , init , rSeed , blockLen, 200 , 20 );

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
