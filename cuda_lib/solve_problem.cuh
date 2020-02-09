#ifndef SOLVE_PROBLEM
#define SOLVE_PROBLEM

/*
  TODO: надо бы разобрраться как работать с профилировщиком.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include "cuda_kernel.cuh"
#include <time.h>
#include <unistd.h>  //  Для функции usleep().
#include <cmath>
#include <iostream>

void solve_problem(const float *MasMatrix, const float *StifMatrix,
    const float *ForceMatrix, float *U, float *V, int n_size, int step) {

  // Парметры решения задачи.
  float tao = 0.001;
	float theta1 = 1, theta2 = 0;

  // Параметры для распаралелливания на видеокарте.
  int block_size = 128; // Количество типа процессоров, выполняющих задачу одновременно.
	int grid_size = 1;  // Блоки объединяются в гриды.

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


  // Считаем время выполнения задачи на CUDA.
  cudaEvent_t start;
  cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  // Выделяю видеопамять под все нужные объекты.
  cudaMalloc((void**)&d_M, size / n_size);
	cudaMalloc((void**)&d_K, size);
	cudaMalloc((void**)&d_F, size / n_size);

	cudaMalloc((void**)&d_u, size / n_size * step);
	cudaMalloc((void**)&d_v, size / n_size * step);
	cudaMalloc((void**)&d_alpha, size / n_size * step);

  cudaMalloc((void**)&d_tmp_force, size / n_size);

  //  Фтксируем время начала ыполнения задачи.
	cudaEventRecord(start, 0);

  // Произвожу копирование требуемой информации в видеопамять.
  cudaMemcpy(d_M, MasMatrix, size / n_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_K, StifMatrix, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, ForceMatrix, size / n_size, cudaMemcpyHostToDevice);

  // TODO: это дело можно вцелом не копировать, т.к. там одни нули.
  // Исключение, если там есть некоторые нненулевые начальные знаечения.
  cudaMemcpy(d_u, U, size / n_size * step, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, V, size / n_size * step, cudaMemcpyHostToDevice);

  // Непосредственно само решение задачи.
  for (int iter; iter<step; ++iter) {
    // Вызов CUDA-kernel, для решения одной итерации задачи.
    solving_system<<<grid_size, block_size>>>(d_M, d_K, d_F, d_u, d_v,
        d_alpha, iter, tao, n_size, d_tmp_force, 0);
  }

  // Копируем данные назад из видеопамяти.
  cudaMemcpy(U, d_u, size / n_size * step, cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_v, size / n_size * step, cudaMemcpyDeviceToHost);

  // Подсчитываем время конца выполнения задачи.
  cudaEventRecord(stop, 0);

  // Считаю суммарное время работы требуемого участка программы.
  // Время - в милисекундах.
  float time = 0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << std::endl;
  std::cout << "GPU computing time: " << time << std::endl;
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
