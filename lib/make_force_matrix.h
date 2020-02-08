#ifndef MAKE_FORCE_MATRIX
#define MAKE_FORCE_MATRIX

/**/

#include "load_mesh_2d.h"
#include "make_local_matrix.h"
#include <cmath>
#include <iostream>

void make_force_matrix(float *ForceMatrix, std::vector<element>& Elements,
    int n_size, int dim, int n_knot) {
  int loc_size = dim * n_knot;
  float *Floc = new float[loc_size];
  float *f_vector = new float[dim];
  f_vector[0] = 0;  // x compoent of vector f
  f_vector[1] = -1;  // y compoent of vector f
  float f = 4000;  // something like f(t)
  for (int i=0; i<n_size; ++i)
      ForceMatrix[i] = 0;

  std::cout << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;
  std::cout << "Start making force matrix..." << std::endl;

  // Предположительно здесь располагается элемент(ы),
  // к которым приложена сила.
  force_matrix_local(Elements[3], Floc, f, f_vector, loc_size);
  assembly_force_matrix(Elements[3], ForceMatrix, Floc, loc_size);

  std::cout << "Force matrix - OK" << std::endl;
  for (int i=0; i<n_size; ++i) {
	  if ( ForceMatrix[i] != 0) {
		  std::cout << "ForceMatrix - Ok" << std::endl;
		  break;
	  }
	  if (i == (n_size - 1)) {
		  std::cout << "Error all zeros force!!" << std::endl;
	  }
  }
  std::cout << "Firce matrix - DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

  delete[] Floc, f_vector;
}

#endif
