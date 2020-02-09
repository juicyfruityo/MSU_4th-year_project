#ifndef LOAD_MESH
#define LOAD_MESH

/*
  Всё это дело работает с c++11, надо добавить
  при компиляции -std=c++11.
  Данная функция загружает из файла объекты
  Nodes и Elements, требуемые для численного метода.
  Пока что данная версия работает только для 2d.

  TODO: сделать поддержку работы 3d.
  Чтобы сделать поддержку 3d достаточно заменить
  node и element, и изменить dim или knot на нужные
  значения.
*/

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <string>
#include <regex>


// IDEA: можно тупо переделать это и element
// под большее количестов узлов и под 3d.
struct node {
  int nid;
  float x, y;

  node(int nid, float x, float y)
      : nid(nid), x(x), y(y)  {}
};

std::vector<node> Nodes;

struct element {
  int eid;
  std::vector<int> num;
  std::vector<node> _node;

  element(int eid, int n1, int n2, int n3, int n4)
         :  eid(eid)
         {
           num.push_back(n1);
           num.push_back(n2);
           num.push_back(n3);
           num.push_back(n4);

           for (int i=0; i<4; ++i) {
               // Ищу нужные node
               int j = 0;
               while (Nodes[j].nid != num[i]) {
                   ++j;
               }
               _node.push_back(Nodes[j]);
               // можно потом совсем убрать Nodes из глобальной областиы
           }
         }
};

std::vector<element> Elements;


int load_mesh(std::string mesh_name, int dimension, int num_knots) {
  std::ifstream f_nodes, f_elements;

  // Это в связи с хранением сеток в отдельной папке.
  std::string mesh_dir = "prepared_meshes/" + mesh_name;
  std::cout << "Start reading files from: " << mesh_dir << std::endl;
  // Т.к. скрипт prepare.py выгружает сетки в подобном виде.
  f_nodes.open(mesh_dir + "/nodes_df.csv");
  f_elements.open(mesh_dir + "/elements_df.csv");

  int dim = dimension; // Размерность задачи, число координат.
  int n_knot = num_knots; // Число узлов на элемент.
  std::cout << "Attention, dimension = " << dim << ". Num of nodes = "
            << n_knot << std::endl;
  // int n1, n2, n3, n4;  // n1-4 были предназначены для Elements.
  // float x, y;  // x, y в Node.
  int nid, eid;
  std::vector<int> n_node(n_knot);  // Номера узлов в элементе (element)
  std::vector<float> coord(dim);  // Координаты узла сетки (node)
  std::string line;
  std::size_t prev=0, next;

  // Считываются nodes.
  while (std::getline(f_nodes, line)) {
      next = line.find(',', prev);
      nid = std::atoi(line.substr(prev, next-prev).c_str());

      for (int i=0; i<dim; ++i) {
          prev = next + 1;
          next = line.find(',', prev);
          coord[i] = std::atof(line.substr(prev, next-prev).c_str());
      }
      prev = 0;

      node tmp(nid, coord[0], coord[1]);
      Nodes.push_back(tmp);
      // std::cout << nid << " " << coord[0] << " " << coord[1] << std::endl;
  }

  std::cout << "Nodes - OK" << std::endl;

  while (std::getline(f_elements, line)) {
      next = line.find(',', prev);
      eid = std::atoi(line.substr(prev, next-prev).c_str());
      prev = next + 1;  // Это надо чтобы не учитывать pid.
      next = line.find(',', prev);

      for (int i=0; i<n_knot; ++i) {
          prev = next + 1;
          next = line.find(',', prev);
          n_node[i] = std::atoi(line.substr(prev, next-prev).c_str());
      }
      prev = 0;

      element tmp(eid, n_node[0], n_node[1], n_node[2], n_node[3]);
  	  Elements.push_back(tmp);
      // std::cout << eid << " " << n_node[0] << " " << n_node[1] << " "
      //           << n_node[2] << " " << n_node[3] << std::endl;
  }

  std::cout << "Elements - OK" << std::endl;
  std::cout << "Loading of mesh DONE" << std::endl;
  for (int i=0; i<25; ++i)
      std::cout << "-";
  std::cout << std::endl;

  f_nodes.close();
  f_elements.close();
  return 0;
}

#endif
