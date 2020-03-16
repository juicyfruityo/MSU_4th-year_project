import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


class PrinterResult:
    def __init__(self, df_elems, df_nodes, df, step=None, NUM=3):
        self.df_elem = df_elems
        self.df_nodes = df_nodes
        self.df = df
        self.step = step
        self.NUM = NUM

    def print_one(self, num, NUM=None, animation=False):
        # Это старая йункция, где я ещё записывал результат по
        # строчкам - узлы, а по столбцам - шаги етрации.
        # Сейчас я сохраняю наоборот, поэтому надо транспонировать.
        # num - видимо номер итерации.
        # NUM - число дополнительных узлов в картинке между узлами в сетке.
        # TODO: надо как-то ускорить, а то для большой сетки - долго.

        df_elem = self.df_elem
        df_nodes = self.df_nodes
        df = self.df

        if not NUM:
            NUM = self.NUM

        for index in df_elem.id:

            node1 = df_elem[df_elem.id == index].node1.values[0]
            node2 = df_elem[df_elem.id == index].node2.values[0]
            node3 = df_elem[df_elem.id == index].node3.values[0]
            node4 = df_elem[df_elem.id == index].node4.values[0]

            x1 = df_nodes[df_nodes.ind == node1].x.values[0]
            x2 = df_nodes[df_nodes.ind == node2].x.values[0]
            y1 = df_nodes[df_nodes.ind == node1].y.values[0]
            y2 = df_nodes[df_nodes.ind == node4].y.values[0]

            test_v = np.array([np.sqrt(df.loc[2*(node1-1), num]**2 + df.loc[2*(node1-1)+1, num]**2),
                               np.sqrt(df.loc[2*(node2-1), num]**2 + df.loc[2*(node2-1)+1, num]**2),
                               np.sqrt(df.loc[2*(node3-1), num]**2 + df.loc[2*(node3-1)+1, num]**2),
                               np.sqrt(df.loc[2*(node4-1), num]**2 + df.loc[2*(node4-1)+1, num]**2)])

            gridX = np.linspace(x2, x1, NUM)
            gridY = np.linspace(y2, y1, NUM)

            meshX, meshY = np.meshgrid(gridX, gridY)

            meshXY = np.dstack((meshX, meshY)).reshape(NUM**2, 2)

            predX = np.linspace(1, -1, NUM)
            predY = np.linspace(1, -1, NUM)
            predX, predY = np.meshgrid(predY, predX)
            predXY = np.dstack((predX, predY)).reshape(NUM**2, 2)
            predict = self.interpolate_test(predXY[:, 0], predXY[:, 1], test_v)

            try:
                pred = np.concatenate((pred, predict), axis=0)
                mesh = np.concatenate((mesh, meshXY), axis=0)
            except Exception as e:
                pred = np.array(predict)
                mesh = np.array(meshXY)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.scatter(mesh[:, 0], mesh[:, 1], c=pred, alpha=0.2)
        plt.axis('equal')
        plt.xlim(2.5, -2.5)
        plt.show()

        if animation is True:
            return mesh, pred

    def update(num, NUM=None):
        mesh, pred = print_one(num, NUM=NUM, animation=True)

        ax.scatter(mesh[:, 0], mesh[:, 1], c=pred, alpha=0.2)
        ax.axis('equal')

        return ax

    def interpolate_test(self, xi, eta, test_v):

        N1 = (1-xi) * (1-eta) / 4.0
        N2 = (1+xi) * (1-eta) / 4.0
        N3 = (1+xi) * (1+eta) / 4.0
        N4 = (1-xi) * (1+eta) / 4.0

        res = N1 * test_v[0] + N2 * test_v[1] \
            + N3 * test_v[2] + N4 * test_v[3]
        return res


def save_animation(printer, NUM=3, name="new_gif"):

    folder_dir = "animation_results/"
    full_dir = folder_dir + name

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def update(num):
        mesh, pred = printer.print_one(num, NUM=NUM, animation=True)

        ax.scatter(mesh[:, 0], mesh[:, 1], c=pred, alpha=0.2)
        ax.axis('equal')

        return ax

    anim = FuncAnimation(fig, update, frames=np.arange(2, 100, 2), interval=250)
    anim.save(full_dir + '.gif', dpi=80, writer='imagemagick')

    print("Animation loaded in {0}".format(folder_dir))
