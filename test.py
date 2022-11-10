import numpy
import matplotlib.pyplot as plt
from bone_marrow.bone_marrow import tsp_solver

points = numpy.array(tsp_solver())
print(points)
edges = numpy.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]])

x = points[:,0].flatten()
y = points[:,1].flatten()

plt.plot(x[edges.T], y[edges.T], linestyle='-', color='y',
         markerfacecolor='red', marker='o')

plt.show()