#ifndef SOLVE_PROBLEM
#define SOLVE_PROBLEM

/**/

void solve_problem(const float *MasMatrix, const float *StifMatrix,
    const float *ForceMatrix, float *U, float *V, int n_size, int step) {

  // Парметры решения задачи.
  loat tao = 0.001;
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



}

#endif
