#ifndef CUDA_KERNEL
#define CUDA_KERNEL

/**/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <cmath>
#include <stdio.h>

__device__ void Ricker_amplitude(float *F, float *force, int iter,
    float tao, float f, int offset) {

	const float pi = 3.141592;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset != 0) {
  		i += offset;
	}

  // f - частота, обратно пропорциональна длине волны.
  // A - амплитуда волны, максимальное удаление от среднего значения.
  // Данное смещение по времени, чтобы сила возрастала от 0.
	// float t = tao * iter - 1.0 / std::sqrt(2 * pi * pi * f * f);
	// float A = (1 - 2 * pi*pi * f*f * t*t) * exp(-pi*pi * f*f * t*t);

  float t0 = 0, t = tao * iter;
  float A = (1 - 2 * pow((pi * f * (t - t0) - pi), 2))
            * exp(-pow((pi * f * (t - t0) - pi), 2));

	force[i] =  F[i] * A;

}

__global__ void solving_system(float *M, float *K, float *F, float *U, float *V,
    float *alpha, int iter, float tao, int n_size, float *tmp_force, int offset=0) {

  // Каждое i отвечает за свой элемент в столбце (свой узел).
	int i = threadIdx.x + blockIdx.x * blockDim.x;

  // Не понимаю зачем это было,
  // вроде чтобы чета распараллелить сильнее.
	// if (offset != 0) {
  //     i += offset;
	// }

  // Начальное ускорение.
	if (iter == 0) {
		  alpha[i] = 0;
	}

  // Считаем перемещения на следующем шаге итерации.
	U[(iter+1)*n_size+i] = U[iter*n_size+i] + V[iter*n_size+i] * tao
      + alpha[iter*n_size+i] * tao * tao / 2;

  // Проводится умножение строки K на вектор U.
  // Именно за счет этого перемещения распространяются
  // на соседние узлы в среде.
	float tmp = 0;
	for (int j=0; j<n_size; ++j) {
      // Мб это не надо делать?.
  		// if (abs(K[i*n_size+j]) <= 0.00001) {
    	// 		K[i*n_size+j] = 0;
  		// }
  		tmp += K[i*n_size+j] * U[(iter+1)*n_size+j];
	}

  // Считаем текущий вектор силы, который изменяется
  // во времени согласно какому-то закону (Например амплитуде Рикера).
  float frequency = 30;  // Было 20
	Ricker_amplitude(F, tmp_force, iter, tao, frequency, offset);

	if (M[i] == 0) {
      // Для избежания конфликта с граничными условиями.
  		alpha[(iter+1)*n_size+i] = 0;
	} else {
      // Решаем систему относительно ускорения.
  		alpha[(iter+1)*n_size+i] = (tmp_force[i] - tmp) / M[i];
	}

  // Делаем обновление вектора скорости.
	V[(iter+1)*n_size+i] = V[iter*n_size+i]
      + tao * (alpha[iter*n_size+i] + alpha[(iter+1)*n_size+i]) / 2;


  // Здесь пишу всю отладку для kernel.
  // if (i == 2*(88 - 1)+1 && iter == 1) {
  //     // printf("%f  ", V[(iter+1)*n_size+i]);
  //     // printf("%f  ", tmp);
  //     for (int j=0; j<n_size; ++j) {
  //         if (K[i*n_size+j] != 0) {
  //             printf("%d :%f   ", j, K[i*n_size+j]);
  //         }
  //     }
  // }

}

#endif
