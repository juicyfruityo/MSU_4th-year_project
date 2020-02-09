#ifndef GPU_SETTINGS
#define GPU_SETTINGS

/**/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

void gpu_settings() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
             prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
             prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n",
             2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
      printf("\n");

      printf("  Max grid size in X : %d\n",
             prop.maxGridSize[0]);
      printf("  Max grid size in Y : %d\n",
             prop.maxGridSize[1]);
      printf("  Max grid size in Z : %d\n",
             prop.maxGridSize[2]);
      printf("\n");

      printf("  Max threads dimention in X : %d\n",
             prop.maxThreadsDim[0]);
      printf("  Max threads dimention in Y : %d\n",
             prop.maxThreadsDim[1]);
      printf("  Max threads dimention in Z : %d\n",
             prop.maxThreadsDim[2]);
      printf("\n");

      printf("  Max threads per block : %d\n",
             prop.maxThreadsPerBlock);
      printf("  Managed memory : %d\n",
            prop.managedMemory);
      printf("  Major compute capability : %d\n",
             prop.major);

      std::cout << std::endl;
      for (int i=0; i<25; ++i)
          std::cout << "-";
      std::cout << std::endl;
  }
}

#endif
