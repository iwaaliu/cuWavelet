#include "cuda_runtime.h"

cudaError_t dwtCuda3d(float *odata, float *idata, const int Nx, const int Ny, const int Nz);

cudaError_t idwtCuda3d(float *odata, float *idata, const int Nx, const int Ny, const int Nz);

cudaError_t dwtCuda3d(float *data, const int Nx, const int Ny, const int Nz);

cudaError_t idwtCuda3d(float *data, const int Nx, const int Ny, const int Nz);

cudaError_t dwtCuda2d(float *odata, float *idata, const int Nx, const int Ny, const int Nz);

cudaError_t idwtCuda2d(float *odata, float *idata, const int Nx, const int Ny, const int Nz);

cudaError_t dwtCuda2d(float *data, const int Nx, const int Ny, const int Nz);

cudaError_t idwtCuda2d(float *data, const int Nx, const int Ny, const int Nz);

cudaError_t dwtCudaTiming(float *odata, float *idata, const int Nx, const int Ny, const int Nz);

int savefile(const char* filename, float *data, int long size);

int savefileBin(const char* filename, float *data, int long size);

float corr2(float* a, float* b, long int size);

float corr2sub(float* a, float* b, long int size, int Nx, int Ny, int n);

float corr2sub(float* a, float* b, long int size, int Nx, int Ny, int n, int flag);

void pearsn(float x[], float y[], unsigned long n, float *r, float *prob, float *z);

void dwtCPU(float *c,  const int Nx, const int Ny, const int Nz);

void setInit(float c[], float a[], int Nx, int Ny, int Nz);

float sumof(float* data,int size);

float meanof(float *a, int size);

float sumof(float* data,long int size);

float meanof(float *a, long int size);