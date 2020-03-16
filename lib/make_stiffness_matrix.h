#ifndef MAKE_STIFFNESS_MATRIX
#define MAKE_STIFFNESS_MATRIX

/*

*/

#include "load_mesh_2d.h"
#include "make_local_matrix.h"
#include <cmath>
#include <iostream>

void make_stiffness_matrix(float *StifMatrix, std::vector<element>& Elements,
    int n_size, int dim, int n_knot, float E, float nu) {

  int loc_size = dim * n_knot;
  float *Kloc = new float[loc_size*loc_size];
  for (int i=0; i<n_size*n_size; ++i)
      StifMatrix[i] = 0;

  std::cout << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;
  std::cout << "Start making stiffness matrix..." << std::endl;

  for (int i=0; i<Elements.size(); ++i) {
      stiffness_matrix_local(Elements[i], Kloc, loc_size, E, nu);
      assembly_one_matrix(Elements[i], StifMatrix, Kloc, n_size, loc_size);

      for(int j=0; j<loc_size*loc_size; ++j) {
          Kloc[j] = 0;
      }
  }

  std::cout << "Stiffness matrix - OK" << std::endl;
  for (int i=0; i<n_size; ++i) {
  	  if (StifMatrix[0*n_size+i] != 0) {
    		  std::cout << "StifMatrix - Ok" << std::endl;
    		  break;
  	  }
  	  if (i == (n_size - 1)) {
        std::cout << "Error all zeros stiff!!" << std::endl;
  	  }
  }
  std::cout << "Stiffness matrix - DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

  delete[] Kloc;
}

#endif
