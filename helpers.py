# importing helpers adds aima to the import path search and
# sets up matplotlib
import os
import sys

# since aima is not distributed as a package, this hack
# is necessary to add it to Python's import search path
sys.path.append(os.path.join(os.getcwd(), 'aima'))

# Hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")

def counter(func):
    """
    A decorator that keeps track of how many times a function is called
    """

    def wrapper(self, *args, **kw_args):
        # all this self nonsense to make it act like the counter is
        # attached to the instance, not the class
        if (not hasattr(wrapper, 'self') or self != wrapper.self):
            wrapper.count = 0

        wrapper.count += 1
        wrapper.self = self

        return func(self, *args, **kw_args)

    wrapper.count = 0
    return wrapper


##
# Edge Matching Helpers
##
import math


def rotate(l, n):  # -n is clockwise
    return l[-n:] + l[:-n]


def print_puzzle(pieces_or_solution):
    pieces = []

    if type(pieces_or_solution) is dict:
        for (_, (piece, rotation)) in sorted(pieces_or_solution.items()):
            pieces.append(rotate(piece, rotation))
    else:
        pieces = pieces_or_solution

    square = math.floor(math.sqrt(len(pieces)))

    for n in range(square):
        for j in [[0], [3, 1], [2]]:
            for i in range(square):
                idx = i+(n*square)
                selection = [str(pieces[idx][k]) for k in j]
                print("{:^9}".format("   ".join(selection)), end='')
            print("")
        print("")


# NQueens
import time
import copy
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from aima.csp import NQueensCSP

import ipywidgets as widgets
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
matplotlib.rcParams['font.family'].append(u'Dejavu Sans')


class Queens(NQueensCSP):
    def __init__(self, size):
        super().__init__(size)
        self.assignment_history = []

    def assign(self, var, val, assignment):
        super().assign(var, val, assignment)
        self.assignment_history.append(copy.deepcopy(assignment))

    def unassign(self, var, assignment):
        super().unassign(var, assignment)
        self.assignment_history.append(copy.deepcopy(assignment))

    def play(self):
        import ipywidgets as widgets
        matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
        matplotlib.rcParams['font.family'].append(u'Dejavu Sans')

        def label_queen_conflicts(assignment, grid):
            ''' Mark grid with queens that are under conflict. '''
            for col, row in assignment.items():  # check each queen for conflict
                row_conflicts = {temp_col: temp_row for temp_col, temp_row in assignment.items()
                                 if temp_row == row and temp_col != col}
                up_conflicts = {temp_col: temp_row for temp_col, temp_row in assignment.items()
                                if temp_row+temp_col == row+col and temp_col != col}
                down_conflicts = {temp_col: temp_row for temp_col, temp_row in assignment.items()
                                  if temp_row-temp_col == row-col and temp_col != col}

                # Now marking the grid.
                for col, row in row_conflicts.items():
                    grid[col][row] = 3
                for col, row in up_conflicts.items():
                    grid[col][row] = 3
                for col, row in down_conflicts.items():
                    grid[col][row] = 3

            return grid

        n = len(self.variables)

        for data in self.assignment_history:
            grid = [[(col+row+1) % 2 for col in range(n)] for row in range(n)]
            # Update grid with conflict labels.
            grid = label_queen_conflicts(data, grid)

            # color map of fixed colors
            cmap = matplotlib.colors.ListedColormap(
                ['white', 'lightsteelblue', 'red'])
            # 0 for white 1 for black 2 onwards for conflict labels (red).
            bounds = [0, 1, 2, 3]
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

            fig = plt.imshow(grid, interpolation='nearest',
                             cmap=cmap, norm=norm)

            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # Place the Queens Unicode Symbol
            for col, row in data.items():
                fig.axes.text(row, col, u"\u265B", va='center',
                              ha='center', family='Dejavu Sans', fontsize=32)

            display.clear_output(wait=True)
            plt.show()
            time.sleep(0.01)


# romania map
import numpy as np
from aima.search import romania_map

romania = romania_map
romania.cities = []
romania.distances = {}

for name in romania.locations.keys():
    romania.cities.append(name)
    romania.distances[name] = {}

for name_1, coordinates_1 in romania.locations.items():
    for name_2, coordinates_2 in romania.locations.items():
        romania.distances[name_1][name_2] = np.linalg.norm(
            [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])
        romania.distances[name_2][name_1] = np.linalg.norm(
            [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])


def nim_minimax_tree(initial, max_move, depth=10):
    # for drawing trees in networkx, which we need for minimax
    def tree_layout(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                    pos=None, parent=None):
        if pos == None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        neighbors = list(G.neighbors(root))
        if parent != None:  # this should be removed for directed graphs.
            # if directed, then parent not in neighbors.
            neighbors.remove(parent)
        if len(neighbors) != 0:
            dx = width/len(neighbors)
            nextx = xcenter - width/2 - dx/2
            for neighbor in neighbors:
                nextx += dx
                pos = tree_layout(G, neighbor, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                    parent=root)
        return pos
            
    i = 1
    b = max_move  # branching factor

    edges = []
    edge_labels = {}
    node_labels = {0: initial}
    node_shapes = {"s": set([]), "v": set([]), "^": set([0])}
    should_continue = True
    
    for d in range(1, depth):
        if not should_continue:
            break
        else:
            should_continue = False
        should_continue = False
        for n in range(b**d):
            child = i+n
            parent = (child-1)//b
            edge = (parent, child)
            edge_label = child % (b) + 1
            
            if parent in node_labels:
                node_label = node_labels[parent] - edge_label

                if node_label >= 0:
                    should_continue = True
                    edges.append(edge)
                    edge_labels[edge] = edge_label
                    node_labels[child] = node_label

                    if node_label == 0:
                        node_shapes["s"].add(child)
                    elif d % 2 == 0:
                        node_shapes["^"].add(child)
                    else:
                        node_shapes["v"].add(child)

        i += b**d

    import networkx as nx
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [12, 8]

    G = nx.Graph()
    G.add_edges_from(edges)
    pos = tree_layout(G, 0)

    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    for shape, nodelist in node_shapes.items():
        labels = {node: node_labels[node] for node in nodelist}
        nx.draw(G, pos, labels=labels, nodelist=nodelist, node_shape=shape,
                node_color='w', edgecolors='black', node_size=1500)
    
    plt.show()
