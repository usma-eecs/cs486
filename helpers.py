# importing helpers adds aima to the import path search and
# sets up matplotlib
import os, sys

# since aima is not distributed as a package, this hack
# is necessary to add it to Python's import search path
sys.path.append(os.path.join(os.getcwd(),'aima'))

# Hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")

def counter(func):
    """
    A decorator that keeps track of how many times a function is called
    """
    def wrapper(self,*args,**kw_args):
        # all this self nonsense to make it act like the counter is
        # attached to the instance, not the class
        if (not hasattr(wrapper,'self') or self != wrapper.self):
            wrapper.count = 0
        
        wrapper.count += 1
        wrapper.self = self
        
        return func(self,*args,**kw_args)
    
    wrapper.count = 0
    return wrapper

##
# Edge Matching Helpers
## 
import math

def rotate(l, n): # -n is clockwise
    return l[-n:] + l[:-n]

def print_puzzle(pieces_or_solution):
    pieces = []
    
    if type(pieces_or_solution) is dict:
        for (_,(piece,rotation)) in sorted(pieces_or_solution.items()):
            pieces.append(rotate(piece,rotation))
    else:
        pieces = pieces_or_solution
        
    square = math.floor(math.sqrt(len(pieces)))
    
    for n in range(square):
        for j in [[0],[3,1],[2]]:
            for i in range(square):
                idx = i+(n*square)
                selection = [str(pieces[idx][k]) for k in j]
                print("{:^9}".format("   ".join(selection)), end='')
            print("")
        print("")