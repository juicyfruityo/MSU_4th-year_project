#ifndef MAKE_DISPLACEMENT_VELOCITY
#define MAKE_DISPLACEMENT_VELOCITY

/**/

void make_displaycement_velocity(float *U, float *V, int n_size, int step) {

  std::cout << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;
  std::cout << "Start making displacement/velocity matrixes..." << std::endl;

  for (int i=0; i<step; ++i) {
  		for (int j=0; j<n_size; ++j) {
    			U[i*n_size+j] = 0;
    			V[i*n_size+j] = 0;

    			if (i == 0) {
      				// Задаю начальные перемещения и скорости в узлах.
      				U[j] = 0;
      				V[j] = 0;
    			}
    	}
	}

  std::cout << "Displacement/velocity matrixes - DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

}

#endif
