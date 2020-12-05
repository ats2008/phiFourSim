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

	sweepMaxCount=eventsRequired*skipSweepCount;
	int N=int(mN/m);
	double w=m;


	//lattice(double mT=1,double wT=1,int nT=120,double dx=1.0,int randseed=0,int skipSweepCount=-1,int writeEventCount=128);
	lattice alat(m,w,N,h,randseed,skipSweepCount,writeEventCount);
	alat.initialize("hot");
	alat.printLattice();

	cout<<"\nDoing simulation for "<<sweepMaxCount<<" sweeps for storing "<<eventsRequired<<" configurations \n";
	cout<<"\n";
	alat.clearFile();
	cout<<"\n";
	alat.takeSweep(sweepMaxCount);
	cout<<"\n";
	cout<<"\n";
	alat.printLattice();
	alat.printToASCII();
	cout<<"\n";
	//alat.takeStride(10);
	//alat.printLattice();
	return 0;
}
