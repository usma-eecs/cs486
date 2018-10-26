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


def nim_decision_tree(objects, moves, depth=10):
    # for drawing trees in networkx, which we need for minimax
    def tree_layout(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,pos=None, parent=None):
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
    b = moves  # branching factor

    edges = []
    edge_labels = {}
    node_labels = {}
    max_utilities = {0: 0}
    min_utilities = {0: 0}
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
            
            if parent in max_utilities:
                if d % 2 == 0:
                    min_utility = min_utilities[parent]
                    max_utility = max_utilities[parent] + edge_label
                else:
                    min_utility = min_utilities[parent] + edge_label
                    max_utility = max_utilities[parent]
                    

                if max_utility + min_utility <= objects:
                    should_continue = True
                    edges.append(edge)
                    edge_labels[edge] = edge_label
                    max_utilities[child] = max_utility
                    min_utilities[child] = min_utility

                    if max_utility + min_utility == objects:
                        if d % 2 == 0:
                            node_labels[child] = max_utility
                        else:
                            node_labels[child] = -min_utility
                            
                        node_shapes["s"].add(child)
                    elif d % 2 == 0:
                        node_shapes["^"].add(child)
                    else:
                        node_shapes["v"].add(child)

        i += b**d

    import networkx as nx
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [16, 8]

    G = nx.Graph()
    G.add_edges_from(reversed(edges))
    pos = tree_layout(G, 0)

    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    for shape, nodelist in node_shapes.items():
        nx.draw(G, pos, labels=node_labels, nodelist=nodelist, node_shape=shape,
                node_color='w', edgecolors='black', node_size=1500)
    
    plt.show()

from aima.mdp import MDP
    
class HiLo(MDP):
    def __init__(self):
        
        rewards = {"bet": -1, "win": 1, "lose": -1}
        actions = {"win": ["draw"],"lose":["exit"],"bet":["draw"]}
        transitions = {"win": {"draw": []}, "lose": {"exit": []},"bet":{"draw":[]}}

        for card in range(1,14):
            rewards[card] = 0
            actions[card] = ["higher","lower"]

            transitions["win"]["draw"].append([1/13,card])
            transitions["bet"]["draw"].append([1/13,card]) 

            transitions[card] = {
                "higher": [[(13-card)/13,"win"], [(card-1)/13,"lose"]],
                "lower":  [[(13-card)/13,"lose"], [(card-1)/13,"win"]]
            }        
        
        MDP.__init__(
            self,
            init="bet", 
            actlist=actions,
            terminals=["lose"], 
            transitions=transitions, 
            reward=rewards, 
            states=None, 
            gamma=1)

    def actions(self,state):
        return self.actlist[state]

    
# The BayesNet implementation from the probability-4e.ipynb notebook
from collections import defaultdict, Counter
import itertools
import math
import random

class BayesNet(object):
    "Bayesian network: a graph of variables connected by parent links."
     
    def __init__(self): 
        self.variables = [] # List of variables, in parent-first topological sort order
        self.lookup = {}    # Mapping of {variable_name: variable} pairs
            
    def add(self, name, parentnames, cpt):
        "Add a new Variable to the BayesNet. Parentnames must have been added previously."
        parents = [self.lookup[name] for name in parentnames]
        var = Variable(name, cpt, parents)
        self.variables.append(var)
        self.lookup[name] = var
        return self
    
class Variable(object):
    "A discrete random variable; conditional on zero or more parent Variables."
    
    def __init__(self, name, cpt, parents=()):
        "A variable has a name, list of parent variables, and a Conditional Probability Table."
        self.__name__ = name
        self.parents  = parents
        self.cpt      = CPTable(cpt, parents)
        self.domain   = set(itertools.chain(*self.cpt.values())) # All the outcomes in the CPT
                
    def __repr__(self): return self.__name__
    
class Factor(dict): "An {outcome: frequency} mapping."

class ProbDist(Factor):
    """A Probability Distribution is an {outcome: probability} mapping. 
    The values are normalized to sum to 1.
    ProbDist(0.75) is an abbreviation for ProbDist({T: 0.75, F: 0.25})."""
    def __init__(self, mapping=(), **kwargs):
        if isinstance(mapping, float):
            mapping = {T: mapping, F: 1 - mapping}
        self.update(mapping, **kwargs)
        normalize(self)
        
class Evidence(dict): 
    "A {variable: value} mapping, describing what we know for sure."
        
class CPTable(dict):
    "A mapping of {row: ProbDist, ...} where each row is a tuple of values of the parent variables."
    
    def __init__(self, mapping, parents=()):
        """Provides two shortcuts for writing a Conditional Probability Table. 
        With no parents, CPTable(dist) means CPTable({(): dist}).
        With one parent, CPTable({val: dist,...}) means CPTable({(val,): dist,...})."""
        if len(parents) == 0 and not (isinstance(mapping, dict) and set(mapping.keys()) == {()}):
            mapping = {(): mapping}
        for (row, dist) in mapping.items():
            if len(parents) == 1 and not isinstance(row, tuple): 
                row = (row,)
            self[row] = ProbDist(dist)

class Bool(int):
    "Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"
    __str__ = __repr__ = lambda self: 'T' if self else 'F'
        
T = Bool(True)
F = Bool(False)

def P(var, evidence={}):
    "The probability distribution for P(variable | evidence), when all parent variables are known (in evidence)."
    row = tuple(evidence[parent] for parent in var.parents)
    return var.cpt[row]

def normalize(dist):
    "Normalize a {key: value} distribution so values sum to 1.0. Mutates dist and returns it."
    total = sum(dist.values())
    for key in dist:
        dist[key] = dist[key] / total
        assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
    return dist

def sample(probdist):
    "Randomly sample an outcome from a probability distribution."
    r = random.random() # r is a random point in the probability distribution
    c = 0.0             # c is the cumulative probability of outcomes seen so far
    for outcome in probdist:
        c += probdist[outcome]
        if r <= c:
            return outcome
        
def globalize(mapping):
    "Given a {name: value} mapping, export all the names to the `globals()` namespace."
    globals().update(mapping)
    
# AIMA calls this enumeration_ask
def query(X, evidence, net):
    "The probability distribution for query variable X in a belief net, given evidence."
    i    = net.variables.index(X) # The index of the query variable X in the row
    dist = defaultdict(float)     # The resulting probability distribution over X
    for (row, p) in joint_distribution(net).items():
        if matches_evidence(row, evidence, net):
            dist[row[i]] += p
    return ProbDist(dist)

def matches_evidence(row, evidence, net):
    "Does the tuple of values for this row agree with the evidence?"
    return all(evidence[v] == row[net.variables.index(v)]
               for v in evidence)

def joint_distribution(net):
    "Given a Bayes net, create the joint distribution over all variables."
    return ProbDist({row: prod(P_xi_given_parents(var, row, net)
                               for var in net.variables)
                     for row in all_rows(net)})

def all_rows(net): return itertools.product(*[var.domain for var in net.variables])

def P_xi_given_parents(var, row, net):
    "The probability that var = xi, given the values in this row."
    dist = P(var, Evidence(zip(net.variables, row)))
    xi = row[net.variables.index(var)]
    return dist[xi]

def prod(numbers):
    "The product of numbers: prod([2, 3, 5]) == 30. Analogous to `sum([2, 3, 5]) == 10`."
    result = 1
    for x in numbers:
        result *= x
    return result