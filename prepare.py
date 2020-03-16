from prepare_mesh_lib.file_reader import FileReader


def main():
    print('ATTENTION: its for 2d only, TODO: add also 3d reading')
    print('-'*25)
    print('Put filename to parse mesh from:')
    filename = input()  # Файл, который надо распарсить. Полное имя.
    # print('Put direction where locate all the meshes:')
    # mesh_dir = input()
    mesh_dir = 'prepared_meshes'  # Папка где будут лежать готовые сетки.
    # print('Put name of directory where all raw meshes:')
    # raw_mesh_dir = input()
    raw_mesh_dir = 'raw_meshes'  # Папка где лежат файлы с неготовыми сетками.
    # print('Put name for new direction:')
    # new_dir = input()
    new_dir = filename[:-2]  # Папка куда сложить результат.

    parser = FileReader(filename, mesh_dir, raw_mesh_dir, new_dir)
    parser.make_directory()

    parser.parse_nodes()
    parser.prepare_nodes()

    parser.parse_elements()
    parser.prepare_elements()

    # parser.make_good()

    print('-' * 25)
    print('Seems like everything nice))')


if __name__ == '__main__':
    main()
