def counter(func):
    """
    A decorator that keeps track of how many times a function is called
    """
    
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return func(*args, **kwargs)

    wrapped.calls = 0
    return wrapped

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