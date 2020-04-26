#ifndef SOLVE_PROBLEM
#define SOLVE_PROBLEM

/*
  TODO: надо бы разобрраться как работать с профилировщиком.
  nvprof - профилировщик для cuda, вроде как.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "cuda_kernel.cuh"
#include <time.h>
#include <unistd.h>  //  Для функции usleep().
#include <cmath>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file,
    int line, bool abort=false) {
  if (code != cudaSuccess) {
      // fprintf(stderr,"GPUassert: %s %s %d", cudaGetErrorString(code), file, line);
      std::cout << "--------------------" << '\n';
      std::cout << "!!!   GPU ERROR   !!!" << '\n';
      std::cout << "Error in line = " << line << '\n';
      std::cout << "Error is " << cudaGetErrorString(code) << '\n';
      std::cout << "--------------------" << '\n';
      if (abort) exit(code);
  } else {
      std::cout << "During GPU work line = " << line << " - OK" << '\n';
  }
}

void solve_problem(const float *MasMatrix, const float *StifMatrix,
    const float *ForceMatrix, float *U, float *V, int n_size, int step) {

  // Парметры решения задачи.
  float tao = 0.001;
	float theta1 = 1, theta2 = 0;

  // Параметры для распаралелливания на видеокарте.
  // Размер блока 1024 - вроде как.
  int block_size = 1024; // Количество типа процессоров, выполняющих задачу одновременно.
	int grid_size = 1 + n_size / 1024;  // Блоки объединяются в гриды.
  int min_grid_size;

  // Более умное определение block/grid_size.
  // вроде как работает.
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
      solving_system, 0, n_size);
  // Round up according to array size
  grid_size = (n_size + block_size - 1) / block_size;

  // Начинаю подготовку для работы с видеокартой.
  // Приставка d_ означает, что эта память предназначена
  // для видеокарты.
	int size = n_size * n_size * sizeof(float);
  float *d_M, *d_K, *d_F;  // Указатели для массы, жесткости, силы.
  float *d_u, *d_v, *d_alpha;  // Указатели для перемещений, скорости, ускорения.
	float *d_tmp_force;  // Указатель для силы, которая будет получаться в задаче.

  // TODO: также надо заботиться вопросом sparse matrix
  // и для вычисленияй на видеокарте, вроде как есть
  // специальный формат в CUDA.

  std::cout << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;
  std::cout << "Start solving problem..." << std::endl;
  std::cout << "        all threads num = " << grid_size*block_size << std::endl;
  std::cout << "        grid_size = " << grid_size << std::endl;
  std::cout << "        block_size = " << block_size << std::endl;


  // Считаем время выполнения задачи на CUDA.
  cudaEvent_t start;
  cudaEvent_t stop;
	gpuErrchk( cudaEventCreate(&start) );
	gpuErrchk( cudaEventCreate(&stop) );

  // Выделяю видеопамять под все нужные объекты.
  gpuErrchk( cudaMalloc((void**)&d_M, size / n_size) );
	gpuErrchk( cudaMalloc((void**)&d_K, size) );
	gpuErrchk( cudaMalloc((void**)&d_F, size / n_size) );

	gpuErrchk( cudaMalloc((void**)&d_u, size / n_size * step) );
	gpuErrchk( cudaMalloc((void**)&d_v, size / n_size * step) );
	gpuErrchk( cudaMalloc((void**)&d_alpha, size / n_size * step) );

  gpuErrchk( cudaMalloc((void**)&d_tmp_force, size / n_size) );

  //  Фтксируем время начала ыполнения задачи.
	gpuErrchk( cudaEventRecord(start, 0) );

  // Произвожу копирование требуемой информации в видеопамять.
  gpuErrchk( cudaMemcpy(d_M, MasMatrix, size / n_size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_K, StifMatrix, size, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(d_F, ForceMatrix, size / n_size, cudaMemcpyHostToDevice) );

  // TODO: это дело можно вцелом не копировать, т.к. там одни нули.
  // Исключение, если там есть некоторые нненулевые начальные знаечения.
  gpuErrchk( cudaMemcpy(d_u, U, size / n_size * step, cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_v, V, size / n_size * step, cudaMemcpyHostToDevice) );

  // Непосредственно само решение задачи.
  // Было <step - мб это как-то влияет.
  for (int iter=0; iter<step-2; ++iter) {
      // Вызов CUDA-kernel, для решения одной итерации задачи.
      solving_system<<<grid_size, block_size>>>(d_M, d_K, d_F, d_u, d_v,
          d_alpha, iter, tao, n_size, d_tmp_force, 0);
  }

  // Копируем данные назад из видеопамяти.
  gpuErrchk( cudaMemcpy(U, d_u, size / n_size * step, cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(V, d_v, size / n_size * step, cudaMemcpyDeviceToHost) );

  // Проверю что сила обнулилась.
  float *tmp_force = new float[n_size];
  gpuErrchk( cudaMemcpy(tmp_force, d_tmp_force, size / n_size, cudaMemcpyDeviceToHost) );
  int flag_tmp = 0;
  for (int i=0; i<n_size; ++i) {
      if (tmp_force[i] != 0) {
          flag_tmp = 1;
          std::cout << "ERROR Force is not ZERO!!!" << '\n';
          std::cout << tmp_force[i] << '\n';
          break;
      }
  }
  if (flag_tmp == 0) {
      std::cout << "Force is ZERO" << '\n';
  }

  // Проверка на работоспособность, что что=-то посчиталось.
  std::cout << "Check for workability of programm: " << '\n';
  int flag = 0;
  for (int j=0; j<step; ++j) {
      for (int i=0; i<n_size; ++i) {
          if (U[j*n_size+i] != 0) {
              std::cout << "------------" << '\n';
              std::cout << "   Nice !!" << '\n';
              flag = 1;
              break;
          }
      }
      if (flag == 1) {
          break;
      }
  }
  if (flag == 0) {
      std::cout << "--------------------------" << '\n';
      std::cout << "    Not Nice, ERROR!!!" << '\n';
  }

  // Подсчитываем время конца выполнения задачи.
  cudaEventRecord(stop, 0);

  // Считаю суммарное время работы требуемого участка программы.
  // Время - в милисекундах.
  float time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << std::endl;
  std::cout << "GPU computing time: " << time << " microseconds" << std::endl;
  // std::cout << std::endl;
  std::cout << "Solving problem - DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

  // Освобождаю всю выделенную видеопамять.
  cudaFree(d_M);
	cudaFree(d_K);
	cudaFree(d_F);

  cudaFree(d_u);
	cudaFree(d_v);
	cudaFree(d_alpha);

	cudaFree(d_tmp_force);
}

#endif
