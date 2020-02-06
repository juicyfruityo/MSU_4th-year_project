#include "lib/load_mesh_2d.h"
#include "lib/make_mass_matrix.h"


int main() {
  // Change name of prepared mesh folder.
  std::string mesh_name = "mesh_6x3";
  // TODO: добавить файл конфигурации сетки, для того, Чтобы
  // можно было удобно выгрудать инфомрацию о сетке.

  // Размерность задачи, количестов узлов на один элемент.
  int dimension = 2, num_knots = 4;

  load_mesh(mesh_name, dimension, num_knots);
  Nodes.clear();

  // Количество узлов * 2 (количество координат (x y), для 3д - 3 координаты).
  // Количестов узлов - количество узлов в Nodes, размер Nodes.
  int n_size = 32 * dimension;
  int step = 100;  // Количество итераций алгоритма.

  // Делаю матрицу масс.
  // Имеет такую размерность, потому что это сама по себе
  // диагональная матрица.
  float *MasMatrix = new float[n_size];
  make_mass_matrix(MasMatrix, Elements, n_size, dimension, num_knots);

  

  delete[] MasMatrix;

  return 0;
}
