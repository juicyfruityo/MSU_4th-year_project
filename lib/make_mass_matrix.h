#ifndef MAKE_MASS_MATRIX
#define MAKE_MASS_MATRIX

/*
  Происзодит сздание матрицы масс с помощью вычисления
  в начале отдельных локальных матриц масс - соответствуюзих элементу
  с последующей агрегацией в общую матрицу масс.
  Матрица масс - диагональная поэтому хранится просто в виде
  одномерного массива.
  Значение в матрице масс зависят от количества прилегающих к узлу
  элементов (например угол - один, граница - 2, внутри - 4).

  Работоспособность проверена - вроде всё окей.
*/

#include "load_mesh_2d.h"
#include "make_local_matrix.h"
#include <cmath>
#include <iostream>

void make_mass_matrix(float *MasMatrix, std::vector<element>& Elements,
    int n_size, int dim, int n_knot) {

  // Делаю матрицу масс - вроде нормально работает.
  int loc_size = (dim * n_knot);  // например 8 для dim=2, n_knot=4.
  float *Mloc = new float[loc_size*loc_size];  // TODO: разобраться откуда 64.
  for (int i=0; i<n_size; ++i) {
      MasMatrix[i] = 0;
  }

  std::cout << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;
  std::cout << "Start making mass matrix..." << std::endl;

  for (int i=0; i<Elements.size(); ++i) {
      mass_matrix_local(Elements[i], Mloc, loc_size);
      assembly_mass_matrix(Elements[i], MasMatrix, Mloc, n_size, loc_size);

      for(int j=0; j<loc_size*loc_size; ++j) {
          Mloc[j] = 0;
      }
  }

  std::cout << "Mass matrix - OK" << std::endl;
  for (int i=0; i<n_size; ++i) {
    if (MasMatrix[i] == 0) {
        std::cout << std::endl;
      	std::cout << "ERROR !!! in mass matrix:  " << i << std::endl;
        std::cout << std::endl;
      	break;
    }
  }
  std::cout << "Mass matrix - DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

  delete[] Mloc;
}

#endif
