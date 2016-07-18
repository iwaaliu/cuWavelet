
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>

//#include "transincl.cu"

#define PI  3.1415926535897932
#define C0  0.4829629131445341
#define C1  0.8365163037378079
#define C2  0.2241438680420134
#define C3 -0.1294095225512604
#define THREAD_NUM 256
#define BLOCK_NUM 2
#define TRANPOSE_BLOCK_DIM 16 //32
#define TRANPOSE_BLOCK_NUM 8
using namespace std;

//cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
//cudaError_t dwtCuda(float *c, float *a, const int Nx, const int Ny, const int Nz)
void daub4(float c[], float a[], unsigned long n, int isign);
void dwtCPU(float *c, const int Nx, const int Ny, const int Nz);
void setInit(float c[], float a[], int Nx, int Ny, int Nz);
__global__ void transpose_kernel(float *odata, const float *idata, int width, int height, int slices, int numBlockY);
__global__ void transpose_Naive(float *odata, const float *idata, int width, int height, int slices, int numBlockY);
void transpose(float* data, const float* idata, const int* dim);
float sumof(float* data,int size);
float meanof(float *a, int size);
float sumof(float* data,long int size);
float meanof(float *a, long int size);

int iDivUp(int a, int b) // Round a / b to nearest higher integer value

	{ return (a % b != 0) ? (a / b + 1) : (a / b); }


__global__ void addKernel(float *c, float *a, float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void initCUDA(int a,int b, int c)
{

	a=blockDim.x;
}

__global__ static void sumOfSquares(float *num, float* result, int Nx, int Ny, int Nz, clock_t* time)
{
	extern __shared__ float shared[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	int DATA_SIZE;
	int i;

	DATA_SIZE=Nx*Ny*Nz;

	if(tid == 0) time[bid] = clock();
	shared[tid] = 0.;

	for(i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
       shared[tid] += num[i] * num[i];
    }

    __syncthreads();

	if(tid < 128) { shared[tid] += shared[tid + 128]; } __syncthreads();
	if(tid < 64) { shared[tid] += shared[tid + 64]; } __syncthreads();
	if(tid < 32) { shared[tid] += shared[tid + 32]; } __syncthreads();
	if(tid < 16) { shared[tid] += shared[tid + 16]; } __syncthreads();
	if(tid < 8) { shared[tid] += shared[tid + 8]; } __syncthreads();
	if(tid < 4) { shared[tid] += shared[tid + 4]; } __syncthreads();
	if(tid < 2) { shared[tid] += shared[tid + 2]; } __syncthreads();
	if(tid < 1) { shared[tid] += shared[tid + 1]; } __syncthreads();

	if(tid == 0) {
		result[bid] = shared[0];
		time[bid + BLOCK_NUM] = clock();
	}
}

__device__ void dwt2dx(float *c, float *a,  int Nx, int Ny, int Nz,int isign, int k)
{
    int tid = threadIdx.x;
   // int size=Nx*Ny*Nz;
   
	int n=Nx;  // &&&&&&&&&&&&
	int nh,nh1;
	if(n<4) return;
	int nn;
	int i,j,kk,iy,iz;
	int i0=k*Nx;//iz*Nz*Ny+iy*Ny;
		if(isign >= 0) { 
			for(nn=n;nn>=4;nn>>=1)
			{
				nh1=(nh=nn>>1)+1;
				for(i=i0,j=i0;j<i0+nn-3;j+=2,i++)
					{
					c[i] = C0*a[j]+C1*a[j+1]+C2*a[j+2]+C3*a[j+3];
					c[i+nh] = C3*a[j]-C2*a[j+1]+C1*a[j+2]-C0*a[j+3];
					}
				c[i]    = C0*a[i0+nn-2]+C1*a[i0+nn-1]+C2*a[i0+0]+C3*a[i0+1];
				c[i+nh] = C3*a[i0+nn-2]-C2*a[i0+nn-1]+C1*a[i0+0]-C0*a[i0+1];
				for(i=i0;i<i0+nn;i++){ a[i]=c[i];}	
			}
		}
		else{
			for(nn=4;nn<=n;nn<<=1){} 
			}
	
}



__global__ void dwt3dx(float *c, float *a,  int Nx, int Ny, int Nz, int isign)
{ //this function is for test only, not actually used because low efficiency
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int i0=Nx*(bid*THREAD_NUM+tid);//iz*Nz*Ny+iy*Ny;
	int n=Nx;
	int i, j;
	int nn, nh,nh1;
	if(n<4) return;
		if(isign >= 0) { 
			for(nn=n;nn>=4;nn>>=1)
			{	nh=nn>>1;
				for(i=i0,j=i0;j<i0+nn-3;j+=2,i++)
					{
					c[i]    = C0*a[j]+C1*a[j+1]+C2*a[j+2]+C3*a[j+3];
					c[i+nh] = C3*a[j]-C2*a[j+1]+C1*a[j+2]-C0*a[j+3];
					}
				c[i]    = C0*a[i0+nn-2]+C1*a[i0+nn-1]+C2*a[i0+0]+C3*a[i0+1];
				c[i+nh] = C3*a[i0+nn-2]-C2*a[i0+nn-1]+C1*a[i0+0]-C0*a[i0+1];
				for(i=i0,j=i0;i<i0+nn;i++,j++){ a[i]=c[j];}	 /// ?? i? j?
			}
		}
		else
		{
			for(nn=4;nn<=n;nn<<=1) 
			{
				nh=nn>>1;
				c[i0]      = C2*a[i0+(nh-1)]+C1*a[i0+(nn-1)]+C0*a[i0]+C3*a[i0+nh];
				c[i0+1] = C3*a[i0+(nh-1)]-C0*a[i0+(nn-1)]+C1*a[i0]-C2*a[i0+nh];
				for(i=i0,j=i0;j<i0+nn-3;j+=2,i++)
					{
					c[j] = C0*a[i]+C1*a[i+nh]+C2*a[i+1]+C3*a[i+(nh+1)]; //j+=1;
					c[j+1] = C3*a[i]-C2*a[i+nh]+C1*a[i+1]-C0*a[i+(nh+1)]; //j+=1;
					}
				for(i=i0,j=i0;i<i0+nn;i++,j++){ a[i]=c[j];}		 /// ?? i? j?
			} 
		}
}

__global__ void dwt3dy(float *c, float *a,  int Nx, int Ny, int Nz, int isign)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int n=Ny;
	int i, j;
	int nn, nh,nh1;
	int nstop; // stopping order, the minimum size to be reduced. if nstop=size(vector), reduce only once.
	if(n<4) return;
	int ndepth=15;

	nstop=ndepth>0?max(4,(n>>(ndepth-1))):4; 
	
	 int I0=bid*THREAD_NUM+tid;
	 //if (I0==0) printf("%d, %d\n", nstop,ndepth);
	 int i0=(I0/Nx)*Nx*Ny+I0%Nx;//iz*Nz*Ny+iy*Ny;

		if(isign >= 0) 
		{ 
			for(nn=n;nn>=nstop;nn>>=1)
			{	nh=nn>>1;
				for(i=i0,j=i0;j<i0+(nn-3)*Nx;j+=2*Nx,i+=Nx)
					{
					c[i]       = C0*a[j]+C1*a[j+1*Nx]+C2*a[j+2*Nx]+C3*a[j+3*Nx];
					c[i+nh*Nx] = C3*a[j]-C2*a[j+1*Nx]+C1*a[j+2*Nx]-C0*a[j+3*Nx];
					}
				c[i]       = C0*a[i0+(nn-2)*Nx]+C1*a[i0+(nn-1)*Nx]+C2*a[i0]+C3*a[i0+1*Nx];
				c[i+nh*Nx] = C3*a[i0+(nn-2)*Nx]-C2*a[i0+(nn-1)*Nx]+C1*a[i0]-C0*a[i0+1*Nx];
				for(i=i0,j=i0;i<i0+nn*Nx;i+=Nx,j+=Nx){ a[i]=c[i];}	 /// ?? i? j?
			}
		}
		else
		{
			for(nn=nstop;nn<=n;nn<<=1)
			{
				nh=nn>>1;
				c[i0]      = C2*a[i0+(nh-1)*Nx]+C1*a[i0+(nn-1)*Nx]+C0*a[i0]+C3*a[i0+nh*Nx];
				c[i0+1*Nx] = C3*a[i0+(nh-1)*Nx]-C0*a[i0+(nn-1)*Nx]+C1*a[i0]-C2*a[i0+nh*Nx];
				for(i=i0,j=i0+2*Nx;i<i0+(nn-1)*Nx;i+=Nx,j+=2*Nx)
					{
					c[j] = C0*a[i]+C1*a[i+nh*Nx]+C2*a[i+1*Nx]+C3*a[i+(nh+1)*Nx]; //j+=1*Nx;
					c[j+Nx] = C3*a[i]-C2*a[i+nh*Nx]+C1*a[i+1*Nx]-C0*a[i+(nh+1)*Nx]; //j+=1*Nx;
					}
				for(i=i0,j=i0;i<i0+nn*Nx;i+=Nx,j+=1*Nx){ a[i]=c[i];}	 /// ?? i? j?
			} 
		}

}

__global__ void dwt3dz(float *c, float *a,  int Nx, int Ny, int Nz, int isign)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int n=Nz;
	int i, j;
	int nn, nh,nh1;
	if(n<4) return;

	 int i0=bid*THREAD_NUM+tid;  // ??? 

	 int ce=Nx*Ny;     // max(i0)=ce-1?

		if(isign >= 0) { 
			for(nn=n;nn>=4;nn>>=1)
			{	nh=nn>>1;
				for(i=i0,j=i0;j<i0+(nn-3)*ce;j+=2*ce,i+=ce)
					{
					c[i]       = C0*a[j]+C1*a[j+1*ce]+C2*a[j+2*ce]+C3*a[j+3*ce];
					c[i+nh*ce] = C3*a[j]-C2*a[j+1*ce]+C1*a[j+2*ce]-C0*a[j+3*ce];
					}
				c[i]       = C0*a[i0+(nn-2)*ce]+C1*a[i0+(nn-1)*ce]+C2*a[i0+0]+C3*a[i0+1*ce];
				c[i+nh*ce] = C3*a[i0+(nn-2)*ce]-C2*a[i0+(nn-1)*ce]+C1*a[i0+0]-C0*a[i0+1*ce];
				for(i=i0,j=i0;i<i0+nn*ce;i+=ce,j+=ce){ a[i]=c[j];}	 /// ?? i? j?
			}
		}
		else{
			for(nn=4;nn<=n;nn<<=1){
				nh=nn>>1;
				c[i0]      = C2*a[i0+(nh-1)*ce]+C1*a[i0+(nn-1)*ce]+C0*a[i0]+C3*a[i0+nh*ce];
				c[i0+1*ce] = C3*a[i0+(nh-1)*ce]-C0*a[i0+(nn-1)*ce]+C1*a[i0]-C2*a[i0+nh*ce];
			  //for(i=i0,j=i0+2*Nx;i<i0+(nn-1)*Nx;i+=Nx,j+=2*Nx)
				for(i=i0,j=i0+2*ce;j<i0+(nn-1)*ce;i+=ce,j+=2*ce)
					{
					c[j] = C0*a[i]+C1*a[i+nh*ce]+C2*a[i+1*ce]+C3*a[i+(nh+1)*ce]; //j+=1*Nx;
					c[j+ce] = C3*a[i]-C2*a[i+nh*ce]+C1*a[i+1*ce]-C0*a[i+(nh+1)*ce]; //j+=1*Nx;
					}
				for(i=i0,j=i0;i<i0+nn*ce;i+=ce,j+=1*ce){ a[i]=c[i];}	 /// ?? i? j?
			} 

			}

}

/*cudaError_t dwtSum(float *yvec, float *xvec, const int Nx, const int Ny, const int Nz)
{
	float sum[BLOCK_NUM];
	
	cudaMemcpy(&sum, result, sizeof(float) * BLOCK_NUM, cudaMemcpyDeviceToHost);
    //cudaMemcpy(&time_used, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
		cudaFree(time);
	float final_sum = 0.0e0;
	for(int i = 0; i < BLOCK_NUM; i++) {
		final_sum += sum[i];
	}
}*/
void timeoutput(clock_t& start, const char* s);

cudaError_t  dwtCuda3d(float *yvec, float *xvec, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size= Nx*Ny*Nz;

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));

    cudaStatus = cudaMemcpy(dev_a, xvec, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	clock_t tic = clock();
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, 1);
	cudaDeviceSynchronize();
	 
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	//prepare to transpose
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);
	//do transpose
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);

	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, 1);
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Ny, Nx, Nz, numBlockY);

	cudaDeviceSynchronize();

	dwt3dz<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, 1);
	cudaDeviceSynchronize();

	/*cudaEvent_t start, stop;
	float time;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start, 0);
	dwt3dz<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, 1);
	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime (&time, start, stop);
	cout<<"GPU time used in z"<<time<<endl;
	cudaEventDestroy (start);
	cudaEventDestroy (stop);*/
	cudaMemcpy(yvec, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaStatus;
}

cudaError_t idwtCuda3d(float *yvec, float *xvec, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size=0;
	size=Nx*Ny*Nz;

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, xvec, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, -1);
	cudaDeviceSynchronize();

	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, -1);
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();

	dwt3dz<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, -1);
	cudaDeviceSynchronize();

	cudaMemcpy(yvec, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaStatus;
}

cudaError_t  dwtCuda3d(float *data, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size = Nx*Ny*Nz;

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
 
    cudaStatus = cudaMemcpy(dev_a, data, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, 1);
	cudaDeviceSynchronize();
	
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);

	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, 1);
	cudaDeviceSynchronize();
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();

	dwt3dz<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, 1);
	cudaDeviceSynchronize();

	cudaMemcpy(data, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaStatus;
}

cudaError_t idwtCuda3d(float *data, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size = Nx*Ny*Nz;

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
 
    cudaStatus = cudaMemcpy(dev_a, data, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, -1);
	cudaDeviceSynchronize();
	
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);

	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, -1);
	cudaDeviceSynchronize();
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();

	dwt3dz<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, -1);
	cudaDeviceSynchronize();

	cudaMemcpy(data, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaStatus;
}

cudaError_t  dwtCuda2d(float *yvec, float *xvec, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size=0;
	size=Nx*Ny*Nz;

    // Allocate GPU buffers for three vectors (two input, one output) .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
 

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, xvec, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	clock_t tic = clock();
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, 1);

	cudaDeviceSynchronize();
	 
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	//prepare to transpose
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);
	//do transpose
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	//transpose_Naive<<<dimGrid, dimBlock>>>(dev_c, dev_a, Nx, Ny, Nz, numBlockY);
//	cudaStatus = cudaMemcpy(dev_a, dev_c, size * sizeof(float), cudaMemcpyDeviceToDevice);
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, 1);
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	//transpose_Naive<<<dimGrid, dimBlock>>>(dev_c, dev_a, Nx, Ny, Nz, numBlockY);
	cudaStatus = cudaMemcpy(dev_c, dev_a, size * sizeof(float), cudaMemcpyDeviceToDevice);
	//timeoutput(tic, "copy back");

	cudaDeviceSynchronize();

	cudaMemcpy(yvec, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
//	timeoutput(tic, "copy out"); 
	return cudaStatus;
}

cudaError_t idwtCuda2d(float *yvec, float *xvec, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size=Nx*Ny*Nz;

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, xvec, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, -1);
	cudaDeviceSynchronize();
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, -1);
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);

	cudaStatus = cudaMemcpy(dev_c, dev_a, size * sizeof(float), cudaMemcpyDeviceToDevice);

	cudaDeviceSynchronize();

	cudaMemcpy(yvec, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaStatus;
}

cudaError_t  dwtCuda2d(float *data, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size = Nx*Ny*Nz;

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
 
    cudaStatus = cudaMemcpy(dev_a, data, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, 1);
	cudaDeviceSynchronize();
	
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);

	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, 1);
	cudaDeviceSynchronize();
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();
	cudaStatus = cudaMemcpy(dev_c, dev_a, size * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(data, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaStatus;
}

cudaError_t idwtCuda2d(float *data, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size = Nx*Ny*Nz;

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
 
    cudaStatus = cudaMemcpy(dev_a, data, size * sizeof(float), cudaMemcpyHostToDevice);
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, -1);
	cudaDeviceSynchronize();
	
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	int dim[3]={Nx,Ny,Nz};
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);

	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();
	dwt3dy<<<nb,nt>>>(dev_c, dev_a, Nx, Ny, Nz, -1);
	cudaDeviceSynchronize();
	transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
	cudaDeviceSynchronize();
	cudaStatus = cudaMemcpy(dev_c, dev_a, size * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(data, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	return cudaStatus;
}

void cudatime_output(cudaEvent_t& start, const char* comment)
{
	cudaEvent_t stop;
	float time;
	cudaEventCreate (&stop);
		cudaEventRecord (stop, 0);
		cudaEventSynchronize (stop);
		cudaEventElapsedTime (&time, start, stop);
		//cout<<"GPU time used in cp-in: "<<time<<" mSecs"<<endl;
		cudaDeviceSynchronize();

		std::cout<<comment<<time<<" mSecs."<<endl;
	start=stop;
}

cudaError_t dwtCudaTiming(float *yvec, float *xvec, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size=0;
	size=Nx*Ny*Nz;

	cudaEvent_t start, stop, begin, end;
	float time, time0;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventCreate (&begin);
	cudaEventCreate (&end);

    // Allocate GPU buffers for three vectors (two input, one output) .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
 

    // Copy input vectors from host memory to GPU buffers.
		cudaEventRecord (start, 0);
		    cudaStatus = cudaMemcpy(dev_a, xvec, size * sizeof(float), cudaMemcpyHostToDevice);
		cudaEventRecord (stop, 0);
		cudaEventSynchronize (stop);
		cudaEventElapsedTime (&time, start, stop);
		//cout<<"GPU time used in cp-in: "<<time<<" mSecs"<<endl;
		cudaDeviceSynchronize();
		cudatime_output(start,"GPU time used in cp-in: ");
	
	int nt=THREAD_NUM ;
	int nb=(Ny*Nz+nt-1)/nt ;

	//y-direction
	cudaEventRecord (start, 0);
		dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, 1);
	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime (&time, start, stop);
	cout<<"GPU time used in dwt-y: "<<time<<" mSecs"<<endl;
	cudaDeviceSynchronize();


	// x-direction
	cudaEventRecord (start, 0);
		dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
															// total number of threads needed is xdim*ydim;
	//prepare to transpose
		int dim[3]={Nx,Ny,Nz};
		int numBlockY=iDivUp(dim[1], dimBlock.y);
		dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);
		cout<<"dimGrid=["<<dimGrid.x<<" "<<dimGrid.y<<" "<<dimGrid.z<<"]"<<endl;
		//do transpose
			cudaEventRecord (begin, 0);
				transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Nx, Ny, Nz, numBlockY);
			cudaEventRecord (end, 0);
			cudaEventSynchronize (end);
			cudaEventElapsedTime (&time0, begin, end);
			cout<<"GPU time used in Transpose->: "<<time0<<" mSecs"<<endl;
			cudaDeviceSynchronize();

			cudaEventRecord (begin, 0);
				dwt3dy<<<nb,nt>>>(dev_c, dev_a, Ny, Nx, Nz, 1);
			cudaEventRecord (end, 0);
			cudaEventSynchronize (end);
			cudaEventElapsedTime (&time0, begin, end);
			cout<<"GPU time used in dwt-y(4 x): "<<time0<<" mSecs"<<endl;
			cudaDeviceSynchronize();

			cudaEventRecord (begin, 0);
				transpose_kernel<<<dimGrid, dimBlock>>>(dev_a, dev_c, Ny, Nx, Nz, numBlockY);
			cudaEventRecord (end, 0);
			cudaEventSynchronize (end);
			cudaEventElapsedTime (&time0, begin, end);
			cout<<"GPU time used in Transpose<-: "<<time0<<" mSecs"<<endl;
			cudaDeviceSynchronize();		

	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime (&time, start, stop);
	cout<<"GPU time used in dwt-x: "<<time<<" mSecs"<<endl;
	cudaDeviceSynchronize();

	//measure dwt-z time
	cudaEventRecord (start, 0);
		dwt3dz<<<nb,nt>>>(dev_a, dev_c, Nx, Ny, Nz, 1);
	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime (&time, start, stop);
	cout<<"GPU time used in dwt-z: "<<time<<" mSecs"<<endl;

	//measure copy-out time
	cudaEventRecord (start, 0);
		cudaMemcpy(yvec, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord (stop, 0);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime (&time, start, stop);
	cout<<"GPU time used in cp-out: "<<time<<" mSecs"<<endl;

	cudaEventDestroy (start);
	cudaEventDestroy (stop);
	cudaEventDestroy (begin);
	cudaEventDestroy (end);
	return cudaStatus;
}


__global__ void calErrKernel(float *c, float *a, float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] - b[i];
}

cudaError_t calErrCuda(float *yvec, float *xvec, const int Nx, const int Ny, const int Nz)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;
	int size=0;
	size=Nx*Ny*Nz;

	return cudaStatus;
}

void timeoutput(clock_t& start, const char* comment)
{
	clock_t end = clock();
	double elapsed_secs = double(end - start);//*1.0e3 / CLOCKS_PER_SEC;
	//std::cout<<"The wavelet transform done."<<endl;
	std::cout<<comment<<"	 "<<elapsed_secs<<" mSecs."<<endl;
	start=end;
}

int savefile(const char* filename, float *data, int long size)
{
	FILE* fp = fopen (filename, "wt");
	printf("Writing file! %d\n", size);
	if (!fp)
		{
			printf("Unable to open file!");
			return 1;
		}
		for ( int i=0; i <size; i++)
		{
			fprintf(fp,"%d\n", int(data[i]));
		}
	//printf("Writing file DONE! %d\n", size);
	fclose(fp);
}

int savefileBin(const char* filename, float *data, int long size)
{
	FILE* fp = fopen (filename, "wb");
	printf("Writing file! %d", size);
	if (!fp)
		{
			printf("Unable to open file!");
			return 1;
		}
		for ( int i=0; i <size; i++)
		{
			fprintf(fp,"%d\n", int(data[i]));
		}
//	printf("Writing file DONE! %d", size);
	fclose(fp);
}

float sumof(float* data,long int size){
	long int i,n;
	float sum=0.0;
	n=size/5;
	for (i=0;i<size;i++)
		{sum+=data[i];}
	return sum;
}

float sumof(float* data,int size){
	int i,n;
	float sum=0.0;
	n=size/5;
	for (i=0;i<size;i++)
		{sum+=data[i];}
	return sum;
}

float meanof(float *a, int size){
	float sum=0;
	sum=sumof(a,size);
	return sum/size;
}

float meanof(float *a,long int size){
	float sum=0;
	sum=sumof(a,size);
	return sum/size;
}
#define TINY 1.0e-20
float corr2(float* a, float* b, long int size)
{
	float corr=0.0;
	float amean,bmean;
	amean=meanof(a,size);
	bmean=meanof(b,size);

	float r=TINY,ru=TINY,rda=TINY,rdb=TINY;
	for (int i=0;i<size;i++){
		float xt=a[i]-amean;
		float yt=b[i]-bmean;
	ru  += xt * yt;
	rda += xt * xt;
	rdb += yt * yt;
	}
	r=sqrt(rda*rdb);
//	printf("%e,%e,%e\n",ru,rda,rdb);
	//if(rda<1e-20 && rdb<1e-20) return 1;

return min(ru/r,1.0);
}

float corr2sub(float* a, float* b, long int size, int Nx, int Ny, int n)
{
	float corr=0.0;
	float amean=0.0,bmean=0.0;
	int n2,i,j,ij;

	n2=n*n;//max(16,3*(n*n/4));
	for (int k=0;k<n2;k++) {
		i=k/n;
		j=k%n;
		int ij=i*Nx+j;
		amean += a[ij];
		bmean += b[ij];
	}

	amean /= n2;
	bmean /= n2;

	float r=TINY,ru=TINY,rda=TINY,rdb=TINY;

	for (int k=0;k<n2;k++) {
		i=k/n;
		j=k%n;
		int ij=i*Nx+j;
		float xt=a[ij]-amean;
		float yt=b[ij]-bmean;
		ru  += xt * yt;
		rda += xt * xt;
		rdb += yt * yt;
	}
	r=sqrt(rda*rdb);
//	printf("%e,%e,%e,%e,%e\n",amean,bmean,ru,rda,rdb);
	//if(rda<1e-20 && rdb<1e-20) return 1;

return min(ru/r,1.0);
}

float corr2sub(float* a, float* b, long int size, int Nx, int Ny, int n, int flag)
{ //if flag is tagged, calculate "_|" shape elements.
	float corr=0.0;
	float amean=0.0,bmean=0.0;
	int n2,i,j,ij;

	n2=max(16,3*(n*n/4));
	for (int k=0;k<n*n;k++) {
		i=k/n;
		j=k%n;
		if(n>4) {				  //n>4: to keep the minimum size block matrix
			if(i>=n/2 || j>=n/2)  //to eliminate left-top quarter block matrix
			{
				int ij=i*Nx+j;
				amean += a[ij];
				bmean += b[ij];
			}
		}
		else{
			int ij=i*Nx+j;		
			amean += a[ij];
			bmean += b[ij];
		}
	}

	amean /= n2;
	bmean /= n2;

	float r=TINY,ru=TINY,rda=TINY,rdb=TINY;

	for (int k=0;k<n*n;k++) {
		i=k/n;
		j=k%n;
		int ij=i*Nx+j;
		if(n>4) 
		{
			if(i>=n/2 || j>=n/2) {
				float xt=a[ij]-amean;
				float yt=b[ij]-bmean;
				ru  += xt * yt;
				rda += xt * xt;
				rdb += yt * yt;
			}
		}
		else{
			float xt=a[ij]-amean;
			float yt=b[ij]-bmean;
			ru  += xt * yt;
			rda += xt * xt;
			rdb += yt * yt;
		}

	}
	r=sqrt(rda*rdb);
	//printf("%e,%e,%e,%e,%e\n",amean,bmean,ru,rda,rdb);
	//if(rda<1e-20 && rdb<1e-20) return 1;

return min(ru/r,1.0);
}

void pearsn(float x[], float y[], unsigned long n, float *r, float *prob,float *z)
{
	float betai(float a, float b, float x);
	float erfcc(float x);
	unsigned long j;
	float yt,xt,t,df;
	float syy=0.0,sxy=0.0,sxx=0.0,ay=0.0,ax=0.0;

	for (j=0;j<n;j++) {
		ax += x[j];
		ay += y[j];
	}
	ax /= n;
	ay /= n;
	for (j=0;j<n;j++) {
		xt=x[j]-ax;
		yt=y[j]-ay;
		sxx += xt*xt;
		syy += yt*yt;
		sxy += xt*yt;
	}
	*r=sxy/sqrt(sxx*syy);
	*z=0.5*log((1.0+(*r)+TINY)/(1.0-(*r)+TINY));
	df=n-2;
	t=(*r)*sqrt(df/((1.0-(*r)+TINY)*(1.0+(*r)+TINY)));
	//*prob=betai(0.5*df,0.5,df/(df+t*t));
}
#undef TINY



void daub4(float *a,  unsigned long n, int isign)
{
	unsigned long nh,nh1,i,j;
	float *c= new float[n] ;
	if (n < 4) return;

	nh1 = (nh = n >> 1);
	if (isign >= 0) {
		for (i=0,j=0;j<n-3;j+=2,i++) {
			c[i]    = C0*a[j]+C1*a[j+1]+C2*a[j+2]+C3*a[j+3];
			c[i+nh] = C3*a[j]-C2*a[j+1]+C1*a[j+2]-C0*a[j+3];
		}
		c[i]    = C0*a[n-2]+C1*a[n-1]+C2*a[0]+C3*a[1];
		c[i+nh] = C3*a[n-2]-C2*a[n-1]+C1*a[0]-C0*a[1];
	} else {}
	for (i=0;i<n;i++) a[i]=c[i];
	delete[] c;
}


void dwtCPU(float *c,  const int Nx, const int Ny, const int Nz)
{
	unsigned long nn;

	int isign=1;
	float *x= new float[Nx] ; 
	//float *y= new float[Nx] ; 
	int n=Nx;
	for(int i=0;i<Ny*Nz;i++)
		{
		for(int j=0;j<Nx;j++) x[j]=c[i*Nx+j];
		if (n < 4) return;
		if (isign >= 0) 
			{
			for (nn=n;nn>=(4);nn>>=1) 
				{daub4(x,nn,1); 
				for (int j=0;j<Nx;j++) c[i*Nx+j]=x[j];
				}
			} 
			else {}
		}


	//delete[] y;

	float *x2= new float[Ny] ; //d
	//float *y2= new float[Ny] ; //d
	n=Ny;
	for(int i=0;i<Nx*Nz;i++)
		{
			for(int j=0;j<Ny;j++) x2[j]=c[j*Ny+i]; //
		if (n < 4) return;
		if (isign >= 0) 
			{
			for (nn=n;nn>=(4);nn>>=1) 
				{daub4(x2,nn,1); 
				for (int j=0;j<Ny;j++) c[j*Ny+i]=x2[j];//
				}
			} 
			else {}
		}

	float *x3= new float[Nz] ; //z
	//float *y3= new float[Nz] ; //z
	n=Nz;
	for(int i=0;i<Nx*Ny;i++)  //
		{
		for(int j=0;j<Nz;j++) x3[j]=c[j*Nx*Ny+i]; //
		if (n < 4) return;
		if (isign >= 0) 
			{
			for (nn=n;nn>=4;nn>>=1) 
				{daub4(x3,nn,1); 
				for (int j=0;j<Nz;j++) c[j*Nx*Ny+i]=x3[j];//
				}
			} 
			else {}
		}
	delete[] x3;
	delete[] x2;
	delete[] x;
	//delete[] y3;
}

void setInit(float c[], float a[], int Nx, int Ny, int Nz)
{
	float x,y,z;
	int zi,yi,xi;
	for(long i=0;i<Nx*Ny*Nz;i++){
		z = PI*(i/Nx/Ny)/180.f;
		zi=i/Nx/Ny;
		yi=(i-Nx*Ny*zi)/Nx;
		y = (PI*yi)/180.0f+PI*0.5f;
		xi=i-zi*Nx*Ny-yi*Nx;
		x = (PI*xi)/180.0+PI*0.25f;
		//a[i]=(PI*i)/180.0f;
		c[i]=sin(x)*sin(y)//*cos(z)
			+sin(10*x)*sin(30*y)//*cos(4*z)
			+sin(50*x)*sin(60*y)*cos(18*z);
		a[i]=c[i];
		//if((i)%65536==0) cout<<x<<" "<<y<<" "<<z<<" "<<i<<"|"<<i/256/256<<endl;
	}
}

__global__ void transpose_Naive(float *odata, const float *idata, int width, int height, int slices, int numBlockY)
{
	unsigned int xIndex = blockIdx.x * TRANPOSE_BLOCK_DIM + threadIdx.x;
	unsigned int blockidy = blockIdx.y%numBlockY;
	unsigned int yIndex =  blockidy* TRANPOSE_BLOCK_DIM + threadIdx.y;
	unsigned int zIndex = blockIdx.y/numBlockY;
	unsigned int slicesize = width * height;

	unsigned int xIndex1 = blockidy * TRANPOSE_BLOCK_DIM + threadIdx.x;
	unsigned int yIndex1 = blockIdx.x * TRANPOSE_BLOCK_DIM + threadIdx.y;

		if((xIndex < width) && (yIndex < height) && zIndex < slices)  // zIndex condition should be satisfied always.
	{
		unsigned int index_in  = zIndex*slicesize + yIndex * width + xIndex;
		unsigned int index_out = zIndex*slicesize + yIndex1 * height + xIndex1;
		odata[index_out] = idata[index_in];
	}
}

/*__global__ void transposeNaive(float *odata, const float *idata)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    odata[x*width + (y+j)] = idata[(y+j)*width + x];
}*/

__global__ void transpose_kernel(float *odata, const float *idata, int width, int height, int slices, int numBlockY)
{
	__shared__ float block[TRANPOSE_BLOCK_DIM][TRANPOSE_BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * TRANPOSE_BLOCK_DIM + threadIdx.x;
	unsigned int blockidy = blockIdx.y%numBlockY;
	unsigned int yIndex =  blockidy* TRANPOSE_BLOCK_DIM + threadIdx.y;
	unsigned int zIndex = blockIdx.y/numBlockY;
	unsigned int slicesize = width * height;
	//printf("%d %d %d %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
	if((xIndex < width) && (yIndex < height) && zIndex < slices)  // zIndex condition should be satisfied always.
	{
		unsigned int index_in = zIndex*slicesize + yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockidy * TRANPOSE_BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * TRANPOSE_BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width) && (zIndex < slices))
	{
		unsigned int index_out = zIndex*slicesize + yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}


#define TILE_DIM 32//32
#define BLOCK_ROWS 8
__global__ void transpose_diagonal(float *odata, const float *idata, int width, int height, int nreps)
{
 __shared__ float tile[TILE_DIM][TILE_DIM+1];
 int blockIdx_x, blockIdx_y;
 // diagonal reordering
 if (width == height) {
 blockIdx_y = blockIdx.x;
 blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
 } else {
 int bid = blockIdx.x + gridDim.x*blockIdx.y;
 blockIdx_y = bid%gridDim.y;
 blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
 }
 int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
 int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
 int index_in = xIndex + (yIndex)*width;
 xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
 yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
 int index_out = xIndex + (yIndex)*height;
 for (int r=0; r < nreps; r++) {
 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 tile[threadIdx.y+i][threadIdx.x] =
 idata[index_in+i*width];
 }

 __syncthreads();

 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 odata[index_out+i*height] =
 tile[threadIdx.x][threadIdx.y+i];
 }
 }
}

__global__ void transposeNaive(float *odata, const float* idata, int width, int height, int nreps)
{
 int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
 int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
 int index_in = xIndex + width * yIndex;
 int index_out = yIndex + height * xIndex;
 for (int r=0; r < nreps; r++) {
 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 odata[index_out+i] = idata[index_in+i*width];
 }
 }
}

__global__ void transposeCoalesced(float *odata, const const float *idata, int width, int height, int nreps)
{
 __shared__ float tile[TILE_DIM][TILE_DIM];
 int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
 int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
 int index_in = xIndex + (yIndex)*width;
 xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
 yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
int index_out = xIndex + (yIndex)*height;
 for (int r=0; r < nreps; r++) {
 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 tile[threadIdx.y+i][threadIdx.x] =
 idata[index_in+i*width];
 }

 __syncthreads();

 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 odata[index_out+i*height] =  tile[threadIdx.x][threadIdx.y+i];
 }
 }
}


__global__ void copySharedMem(float *odata, const float *idata, int width, int height, int nreps)
{
 __shared__ float tile[TILE_DIM][TILE_DIM];
 int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
 int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;

 int index = xIndex + width*yIndex;
 for (int r=0; r < nreps; r++) {
 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 tile[threadIdx.y+i][threadIdx.x] =
 idata[index+i*width];
 }

 __syncthreads();

 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 odata[index+i*width] =
 tile[threadIdx.y+i][threadIdx.x];
 }
 }
}

__global__ void transposeNoBankConflicts(float *odata, const float *idata, int width, int height, int nreps)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int r=0; r < nreps; r++)
    {
        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }

        __syncthreads();

        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
        }
    }
}

__global__ void transposeFineGrained(float *odata, const float *idata, int width, int height, int nreps)
{
 __shared__ float block[TILE_DIM][TILE_DIM+1];
 int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
 int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
 int index = xIndex + (yIndex)*width;
 for (int r=0; r<nreps; r++) {
 for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
 block[threadIdx.y+i][threadIdx.x] =
 idata[index+i*width];
 }

 __syncthreads();
 for (int i=0; i < TILE_DIM; i += BLOCK_ROWS) {
 odata[index+i*height] =
 block[threadIdx.x][threadIdx.y+i];
 }
 }
}
__global__ void transposeCoarseGrained(float *odata, const float *idata, int width, int height, int nreps)
{
 __shared__ float block[TILE_DIM][TILE_DIM+1];
 int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
 int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
 int index_in = xIndex + (yIndex)*width;
 xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
 yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
 int index_out = xIndex + (yIndex)*height;
 for (int r=0; r<nreps; r++) {
 for (int i=0; i<TILE_DIM; i += BLOCK_ROWS) {
 block[threadIdx.y+i][threadIdx.x] =
 idata[index_in+i*width];
 }

 __syncthreads();
 for (int i=0; i<TILE_DIM; i += BLOCK_ROWS) {
 odata[index_out+i*height] =  block[threadIdx.y+i][threadIdx.x];
 }
 }
}


// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(float *gold, const float *idata, const  int size_x, const  int size_y)
{
    for (int y = 0; y < size_y; ++y)
    {
        for (int x = 0; x < size_x; ++x)
        {
            gold[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}


void transpose(float* odata, const float* idata, const int* dim)
{
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
	// total number of threads needed is xdim*ydim;
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);
	//transpose_kernel<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2], numBlockY);
	//transpose_diagonal<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2]);
	//transposeNaive<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2]);
	//transposeCoalesced<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2]);
	//transposeNoBankConflicts<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2]);
	//transposeFineGrained<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2]);
	transposeCoarseGrained<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2]);

}

/*void transpose(float* odata, const float* idata, const int* dim)
{
	dim3 dimBlock(TRANPOSE_BLOCK_DIM, TRANPOSE_BLOCK_DIM);  // num of threads per block is BLOCK_SIZE*BLOCK_SIZE
	// total number of threads needed is xdim*ydim;
	int numBlockY=iDivUp(dim[1], dimBlock.y);
	dim3 dimGrid(iDivUp(dim[0], dimBlock.x), numBlockY*dim[2]);
	transpose_kernel<<<dimGrid, dimBlock>>>(odata,  idata, dim[0], dim[1], dim[2], numBlockY);
}*/

/*__global__ void dwt2yshared(float *odata, const const float *idata, int width, int height, int nreps)
{
 __shared__ float tile[TILE_DIM*TILE_DIM];
 int xIndex = blockIdx.x*TILE_DIM + threadIdx.x;
 int yIndex = blockIdx.y*TILE_DIM + threadIdx.y;
 int index_in = xIndex + (yIndex)*width;
 xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
 yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
int index_out = xIndex + (yIndex)*height;
 for (int r=0; r < nreps; r++) {
 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 tile[threadIdx.y+i][threadIdx.x] =
 idata[index_in+i*width];
 }

 __syncthreads();

 for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
 odata[index_out+i*height] =  tile[threadIdx.x][threadIdx.y+i];
 }
 }
}*/



__global__ void dwt2yshared(float *odata, float *idata,  int Nx, int Ny, int Nz, int isign)
{
	 __shared__ float a[TILE_DIM*TILE_DIM];
	 __shared__ float c[TILE_DIM*TILE_DIM];
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int n=Ny;
	int i, j;
	int nn, nh,nh1;
	int nstop; // stopping order, the minimum size to be reduced. if nstop=size(vector), reduce only once.
	if(n<4) return;
	int ndepth=15;

	nstop=ndepth>0?max(4,(n>>(ndepth-1))):4; 
	
	 int I0=bid*THREAD_NUM+tid;
	 //if (I0==0) printf("%d, %d\n", nstop,ndepth);
	 int i0=(I0/Nx)*Nx*Ny+I0%Nx;//iz*Nz*Ny+iy*Ny;

		if(isign >= 0) 
		{ 
			for(nn=n;nn>=nstop;nn>>=1)
			{	nh=nn>>1;
				for(i=i0,j=i0;j<i0+(nn-3)*Nx;j+=2*Nx,i+=Nx)
					{
					c[i]       = C0*a[j]+C1*a[j+1*Nx]+C2*a[j+2*Nx]+C3*a[j+3*Nx];
					c[i+nh*Nx] = C3*a[j]-C2*a[j+1*Nx]+C1*a[j+2*Nx]-C0*a[j+3*Nx];
					}
				c[i]       = C0*a[i0+(nn-2)*Nx]+C1*a[i0+(nn-1)*Nx]+C2*a[i0]+C3*a[i0+1*Nx];
				c[i+nh*Nx] = C3*a[i0+(nn-2)*Nx]-C2*a[i0+(nn-1)*Nx]+C1*a[i0]-C0*a[i0+1*Nx];
				for(i=i0,j=i0;i<i0+nn*Nx;i+=Nx,j+=Nx){ a[i]=c[i];}	 /// ?? i? j?
			}
		}
		else
		{
			for(nn=nstop;nn<=n;nn<<=1)
			{
				nh=nn>>1;
				c[i0]      = C2*a[i0+(nh-1)*Nx]+C1*a[i0+(nn-1)*Nx]+C0*a[i0]+C3*a[i0+nh*Nx];
				c[i0+1*Nx] = C3*a[i0+(nh-1)*Nx]-C0*a[i0+(nn-1)*Nx]+C1*a[i0]-C2*a[i0+nh*Nx];
				for(i=i0,j=i0+2*Nx;i<i0+(nn-1)*Nx;i+=Nx,j+=2*Nx)
					{
					c[j] = C0*a[i]+C1*a[i+nh*Nx]+C2*a[i+1*Nx]+C3*a[i+(nh+1)*Nx]; //j+=1*Nx;
					c[j+Nx] = C3*a[i]-C2*a[i+nh*Nx]+C1*a[i+1*Nx]-C0*a[i+(nh+1)*Nx]; //j+=1*Nx;
					}
				for(i=i0,j=i0;i<i0+nn*Nx;i+=Nx,j+=1*Nx){ a[i]=c[i];}	 /// ?? i? j?
			} 
		}

}