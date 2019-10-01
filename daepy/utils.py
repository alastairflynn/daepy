import numpy as np
from collocation import CollocationSolution
import dill

def load_data(filename):
    with open(filename, 'rb') as file:
        data = dill.load(file)
    solution = data['solution']
    scale = data['scale']
    try:
        transform = data['transform']
    except:
        transform = None
    points = solution.collocation_points
    initial = solution.eval(points)
    return solution, initial, points, transform, scale

def save_data(savename, y, transform, scale):
    with open('data/' + savename + '.dill', 'wb') as file:
        dill.dump({'solution':y, 'transform':transform, 'scale':scale}, file)
