#include "lib/load_mesh_2d.h"
#include "lib/make_mass_matrix.h"
#include "lib/make_stiffness_matrix.h"
#include "lib/make_force_matrix.h"
#include "lib/make_border_condition.h"
#include "lib/make_displacement_velocity.h"
#include "cuda_lib/solve_problem.cuh"
#include "lib/save_results.h"
#include "cuda_lib/gpu_settings.cuh"

/*
  Компилировать всё это дело надо -std=c++11,
  компиляртор для cuda - nvcc.

  Преобразование от nid (номер узла) к соответсвующему ему
  значению строки/столбца определяется по формуле:
      для координаты X - 2 * (nid - 1)
      для координаты Y - 2 * (nid - 1) + 1


*/


int main() {
  gpu_settings();

  // Change name of prepared mesh folder.
  std::string mesh_name = "test_lemb_v2_4141nodes";
  // std::string mesh_name;

  // std::cout << std::endl << "Put name of folder:" << std::endl;
  // std::cin >> mesh_name;  // Только имя папки.
  // std::cout << std::endl;
  // TODO: добавить файл конфигурации сетки, для того, Чтобы
  // можно было удобно выгрудать инфомрацию о сетке.

  // Размерность задачи, количестов узлов на один элемент.
  std::cout << "!!! Don't forget change program parameters !!!" << '\n';
  std::cout << "        n_size, step;\n        ro, E, nu, mu;\n        "
            << "force, vector, Elements, etc...;\n        border condition;\n"
            << "        additional info, folder name;" << '\n';
  std::cout << std::endl;

  int dimension = 2, num_knots = 4;

  load_mesh(mesh_name, dimension, num_knots);
  // Nodes.clear();  // Вроде как ещё понадобится для ганичных условий.

  // Количество узлов * 2 (количество координат (x y), для 3д - 3 координаты).
  // Количестов узлов - количество узлов в Nodes, размер Nodes.
  int n_size = 4141 * dimension;
  int step = 1000;  // Количество итераций алгоритма.

  // Делаю матрицу масс.
  // Имеет такую размерность, потому что это сама по себе
  // диагональная матрица.
  float *MasMatrix = new float[n_size];
  make_mass_matrix(MasMatrix, Elements, n_size, dimension, num_knots);

  // Делаю матрицу жесткости.
  // Менять свойства среды надо внутри - надобы это исправить.
  // TODO: переделать на спарс матрицу.
  // Надоы бы как-то проверить четче как работает её подсчет
  // + разобрать вцелом ч там да как под капотом
  // + надо привести параметризацию задачи в божеский вид
  // т.е. чтобы в main задавать модуль Юнга и параметры Ламэ.
  // Это слабое место, при подсчете B могут возникнуть ошибки.
  float E = 10000000;
  float nu = 0.0;
  float *StifMatrix = new float[n_size*n_size];
  make_stiffness_matrix(StifMatrix, Elements, n_size, dimension, num_knots, E, nu);

  // Делаю матрицу сил.
  // TODO: опять же нормально разобраться, что там внутри
  // плюс, так же привести параметризацию в нормальный вид.
  // Аналогично надо сюда забабахать sparse матрицу.
  float *ForceMatrix = new float[n_size];
  make_force_matrix(ForceMatrix, Elements, n_size, dimension, num_knots);

  // Проверяю работоспособнось - сила должна быть на
  // узлах, соответсвующих элементу, к которому приложена
  // номер узла == порядковый номер i+1 (т.к. нумерация с 0).
  // for (int i=0; i<n_size; ++i) {
  //     std::cout << ForceMatrix[i] << " " << i << " " << int(i/2) << std::endl;
  // }

  // Требуется задать граничные условия, с помощью изменения
  // матрицы жесткости (вроде как надо занулить соответсвующий
  // столбец и строчку).
  // TODO: разобраться как работают в данном случае граничные условия.
  float y_bord = 100; // было и там и там 10
  float x_bord = 100;

  // TODO: поэкспериментировать, все ли будет ок, с такими гр. условиями.
  make_border_condition(Nodes, MasMatrix, StifMatrix, ForceMatrix,
      n_size, y_bord, x_bord);

  // Проверю работу граничных условий.
  // for (int i=0; i<n_size; ++i) {
      // std::cout << MasMatrix[i] << " " << int(i/2) << std::endl;
      // std::cout << ForceMatrix[i] << " " << int(i/2) << std::endl;
  // }

  // Создаю матрицы в которых буду хранить полученные
  // перемещения и скорости в узлах сетки.
  float *U = new float[step*n_size];  // Перемезения в узлах.
	float *V = new float[step*n_size];  // Скорость в узлах.

  // По строчкам - значения перемещений в узлах на каждом
  // конкретном шаге итерации.
  // Строчки отвечают за шаг алгоритма, столбцы - за узел.
  // Менять начальные перемещения/скорости внутри.
  // Если всё ок - сюда тоже надо sparse запилить.
  make_displaycement_velocity(U, V, n_size, step);

  // TODO: сделать вывод параметров видекарты, которая есть в системе.

  // Здесь происходит непосредственное решение задачи.
  // TODO: аналогично всему, надо подстроить под масштабируемость.
  // + так же надо бы нормальным образом подсчитывать время выполнения.
  // + надо добавить отслеживание ошибок при выполнении.
  // + надо нормально сделать параметризацию задачи.
  // Параметры решения задачи меняются внутри (пока что - надуюсь).
  solve_problem(MasMatrix, StifMatrix, ForceMatrix, U, V, n_size, step);

  // Сохранение матрицы перемещений и скорости для последующего анализа.
  std::string addition_info = "test_v5";
  save_results(U, V, mesh_name, n_size, step, addition_info);

  // TODO: ещё можно удалить Elements и Nodes, если это как-то значимо влияет.

  delete[] MasMatrix;
  delete[] StifMatrix;
  delete[] ForceMatrix;
  delete[] U;
  delete[] V;

  return 0;
}
