#ifndef SAVE_RESULTS
#define SAVE_RESULTS

/**/

#include <unistd.h>  // Для cwd;
#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iostream>

void save_results(float* U, float* V, std::string mesh_name,
    int n_size, int step) {

  // Храню результаты в отдельной папке с названием,
  // таким же как название сетки.
  std::string res_dir = "raw_results/" + mesh_name;
  const char *new_dir = res_dir.c_str();

  std::cout << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;
  std::cout << "Write raw results in " << res_dir << "..." << std::endl;

  // Создаю папку, куда солью файлы с результатми.
  mkdir(new_dir, 0777);

  std::ofstream f_displacements(res_dir + "/displacments.csv");
  std::ofstream f_velocities(res_dir + "/velocities.csv");

  for (int i=0; i<step; ++i) {
      for (int j=0; j<n_size; ++j) {
          f_displacements << U[i*n_size+j];
          f_velocities << V[i*n_size+j];

          if (j < n_size - 1) {
              f_displacements << ",";
              f_velocities << ",";
          } else {
              f_displacements << "\n";
              f_velocities << "\n";
          }
      }
  }

  std::cout << "Writing raw results - DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

  f_displacements.close();
  f_velocities.close();

}

#endif
