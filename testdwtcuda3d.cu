#include "dwtcuda3d.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;




int main()
{

	
	cudaDeviceProp  prop;

	int count;
	cudaGetDeviceCount(&count);
	for (int i = 0; i< count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("   --- General Information for device %d ---\n", i);
		printf("Name:  %s\n", prop.name);
		printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf("Clock rate:  %d\n", prop.clockRate);
		printf("Device copy overlap:  ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execution timeout :  ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem:  %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem:  %ld\n", prop.totalConstMem);
		printf("Max mem pitch:  %ld\n", prop.memPitch);
		printf("Texture Alignment:  %ld\n", prop.textureAlignment);

		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count:  %d\n",
			prop.multiProcessorCount);
		printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp:  %d\n", prop.regsPerBlock);
		printf("Threads in warp:  %d\n", prop.warpSize);
		printf("Max threads per block:  %d\n",
			prop.maxThreadsPerBlock);
		printf("Max thread dimensions:  (%d, %d, %d)\n",
			prop.maxThreadsDim[0], prop.maxThreadsDim[1],
			prop.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n",
			prop.maxGridSize[0], prop.maxGridSize[1],
			prop.maxGridSize[2]);
		printf("\n");
	}

	const int Nx=512;
	const int Ny=512;
	const int Nz=512;

	long size=Nx*Ny*Nz;
	long figsize=Nx*Ny;

	std::vector<float> xcor(size, 1.0) ; 
	std::vector<float> yval(size, 1.0) ;
	std::vector<float> figa(figsize, 0.0) ;
	std::vector<float> figb(figsize, 0.0) ;

#if 1  //GPU/CPU timing test

    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    printf ( "Time&date: %s", ctime (&rawtime) );

	cout<<"Dimensions:"<<" "<<Nx<<" "<<Ny<<" "<<Nz<<endl;
	cout<<"Initial set."<<endl;
	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);
	cudaFree(0); // force CUDA to initialize by accessing CUDA context

	std::cout<<"Doing wavelet transform (forward)..."<<endl;
	clock_t begin = clock();
	dwtCudaTiming(yval.data(), yval.data(), Nx, Ny, Nz);

	clock_t end = clock();
	double elapsed_secs11 = double(end - begin) / CLOCKS_PER_SEC;
	std::cout<<"The wavelet transform (GPU) done."<<endl;
	std::cout<<"Time used in GPU: "<<elapsed_secs11<<" Secs."<<endl;



	std::cout<< "Doing wavelet transform (CPU) ..."<<endl;
	begin = clock();
	dwtCPU(yval.data(), Nx, Ny, Nz);
	end = clock();
	double elapsed_secs12 = double(end - begin) / CLOCKS_PER_SEC;
	std::cout<<"The wavelet transform (CPU) done."<<endl;
	std::cout<<"Time used in CPU: "<<elapsed_secs12<<" Secs."<<endl;
	std::cout<<"CPU/GPU: "<<elapsed_secs12/elapsed_secs11<<" times speedup."<<endl<<endl;

	cout<<"Press enter to exit."<<endl;
	cin.get();
    return 0;

#endif


	FILE* fp;
	FILE* fp1,*fpx;

#if 0	
	fp = fopen ("MyFile.txt", "rt");
	fp1= fopen ("OutFile.txt", "wt");
	fpx= fopen ("xFile.txt", "wt");

	if (!fp)
		{
			printf("Unable to open file!");
			return 1;
		}
		for ( int i=0; i <size; i++)
		{
			fscanf(fp,"%f",&yval[i]);
			xcor[i]=yval[i];//yval[i]=1.0;
		    //fprintf(fp1,"%d\n", int(yval[i]));
			//fprintf(fpx,"%d\n", 256-int(xcor[i]));
		}

	fclose(fp);
	fclose(fp1);
	fclose(fpx);
#endif

#if 1	
char cstr[256];
//char dir[]="data\\";
char dir[]="data2\\";
for (int i = 0;i<Nz;i++){
	//sprintf(cstr,"%sct.%d.txt",dir,i);
	sprintf(cstr,"%sshift.%d.txt",dir,i);
	fp = fopen (cstr, "rt");
	if (!fp)
		{
			printf("Unable to open file: %s!",cstr);
			return 1;
		}
		for ( int j=0; j <figsize; j++)
		{	
			long k=i*figsize+j;
			fscanf(fp,"%f",&yval[k]);
			xcor[k]=yval[k];
		}
	fclose(fp);
}

#endif


	cout<<"Initial set."<<endl;
	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaSetDevice(0);
	cudaFree(0); // force CUDA to initialize by accessing CUDA context

	std::cout<<"Doing wavelet transform (forward)..."<<endl;
	clock_t begin0 = clock();
	dwtCuda2d(yval.data(), yval.data(), Nx, Ny, Nz);
	
	//std::cout<<"Doing wavelet transform (inverse)..."<<endl;

	//idwtCuda2d(yval.data(), yval.data(), Nx, Ny, Nz);
	clock_t end0 = clock();
	double elapsed_secs = double(end0 - begin0) / CLOCKS_PER_SEC;
	std::cout<<"The wavelet transform done."<<endl;
	std::cout<<"Time used in GPU: "<<elapsed_secs<<" Secs."<<endl;




	savefile("OutFile1.txt",yval.data(),size);
	
	float err2=0;
	for (long i=0;i<figsize;i++){
	err2+=(yval[i]-xcor[i])*(yval[i]-xcor[i]);
	}

	cout<<endl<<setprecision(20)<<"DWT ERR is:"<<err2<<endl;
	//return 0;

	cout<<"after DWT:"<<endl;
	int n=2;

	//cout<<setprecision(15)<<"corr:"<<corr<<endl;
	fp = fopen("corr.txt","wt");
	int nc=0;//min(Nz-1,60);
	for (int i=0;i<Nz;i++){
		cout<<endl<<"i="<<i<<":  ";
		n=2;
		float corr=corr2sub(yval.data()+nc*figsize,yval.data()+i*figsize,figsize,Nx,Ny,Nx); 
		fprintf(fp,"%f ",corr);
		for (int j=0;j<8;j++){
			corr=corr2sub(yval.data()+nc*figsize,yval.data()+i*figsize,figsize,Nx,Ny,n<<=1,1);
			//cout<<setprecision(5)<<"n="<<n<<" corr:"<<corr<<"  ";//<<endl;
			fprintf(fp,"%f ",corr);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

#if 0
	for (int i=0;i<Nz;i++){
		corr=corr2sub(yval.data()+i*figsize,yval.data()+((i+1)%3)*figsize,figsize,Nx,Ny,Nx);
		cout<<setprecision(15)<<"corr:"<<corr<<endl;
		corr=corr2sub(yval.data()+i*figsize,yval.data()+((i+1)%3)*figsize,figsize,Nx,Ny,Nx,1);
		//corr=corr2sub(yval.data(),xcor.data(),figsize,Nx,Ny,4);
		cout<<setprecision(15)<<"corr:"<<corr<<endl;
	}
#endif
	return 0;

	savefile("OutFile.txt",yval.data(),size);
	//savefileBin("OutFile.bin",yval.data(),size);

	cudaDeviceReset();

 return 0;

	std::cout<< "Doing wavelet transform (CPU) ..."<<endl;
	begin = clock();
	//dwtCPU(yval.data(), Nx, Ny, Nz);
	end = clock();
	double elapsed_secs0 = double(end - begin) / CLOCKS_PER_SEC;
	std::cout<<"The wavelet transform done (CPU)."<<endl;
	std::cout<<"Time used in CPU: "<<elapsed_secs0<<" Secs."<<endl;
	std::cout<<"Time ratio CPU/GPU: "<<elapsed_secs0/elapsed_secs<<" times speedup."<<endl<<endl;
	for(int i =0;i<9;i++) cout<<yval[i]<<" ";cout<<endl;
	//for(int i =0;i<9;i++) cout<<setprecision(3) <<yval1[i+Nx-5]<<" ";

	//cin.get();
    return 0;
	
}

