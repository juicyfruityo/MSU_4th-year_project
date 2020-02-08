#ifndef MAKE_BORDER_CONDITION
#define MAKE_BORDER_CONDITION

/**/

#include "load_mesh_2d.h"
#include <cmath>

void make_border_condition(std::vector<node>& Nodes, float *MasMatrix,
    float *StifMatrix, float *ForceMatrix, int n_size, int y_bord, int x_bord) {

  std::cout << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;
  std::cout << "Start making border conditon..." << std::endl;

  for (int i=0; i<Nodes.size(); ++i) {
      if (abs(Nodes[i].x) == x_bord || abs(Nodes[i].y) == y_bord) {
          int k_x = 2 * (Nodes[i].nid - 1);  // Было только это.
          // Добавлю ещё для y.
          int k_y = 2 * (Nodes[i].nid - 1) + 1;
          ForceMatrix[k_x] = 0;
    			MasMatrix[k_x] = 0;

          ForceMatrix[k_y] = 0;
    			MasMatrix[k_y] = 0;

          for (int j=0; j<n_size; ++j) {
              StifMatrix[k_x*n_size+j] = 0;
              StifMatrix[j*n_size+k_x] = 0;
              if (k_x == j) {
                  StifMatrix[k_x*n_size+j] = 1;
              }

              StifMatrix[k_y*n_size+j] = 0;
              StifMatrix[j*n_size+k_y] = 0;
              if (k_y == j) {
                  StifMatrix[k_y*n_size+j] = 1;
              }
          }
      }
  }

  std::cout << "Border condition - DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

}

#endif
