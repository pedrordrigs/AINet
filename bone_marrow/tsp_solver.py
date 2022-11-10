import random
from opentsp.objects import Generator

def tsp_generator(len):
    gen = Generator()
    instance = gen.new_instance(len, source='seed', seed=1234)
    instance.view(nodes=True, edges=True)
    return instance

def gen_libraries(instance, num_libraries, library_size):
    x = instance.x_values
    y = instance.y_values
    gene_libraries = []
    for i in range(num_libraries):
        coord = []
        for i in range(len(x)):
            coord.append([x[i],y[i]])

        gene_lib = []

        for j in range(library_size):
            l = 0
            n = 0
            while(l == n):
                l = random.randint(0, len(coord)-1)
                n = random.randint(0, len(coord)-1)
            gene_lib.append(coord[l] + coord[n])

        gene_libraries.append(gene_lib)
    return gene_libraries

def gen_chromossome(gene_libraries,num_libraries):
    selected_genes = []
    for i in range(num_libraries):
        selected_genes.append(gene_libraries[i][random.randint(0, len(gene_libraries[i])-1)])
    return selected_genes

def repair_chromossome(chromossome, instance, num_libraries):
    repaired = []
    x = instance.x_values
    y = instance.y_values
    for i in range(num_libraries):
        coord = []
        for i in range(len(x)):
            coord.append([x[i],y[i]])

    for i in range(len(chromossome)):
        for j in range(0, len(chromossome[i]), 2):
            repaired.append([chromossome[i][j], chromossome[i][j+1]])

    print(repaired)
    replace = [[]]
    for i in range (len(repaired)):
        for j in range (len(repaired)):
            if(repaired[i] == repaired[j] and i != j):
                check = 1
                m = 0
                while(check != 0):
                    check = 0
                    replace[0] = (coord[m])
                    for n in range(len(repaired)):
                        if(repaired[n] == replace[0]):
                            check += 1
                    m += 1
                repaired[j] = replace[0]

    print(repaired)
    return(repaired)



def fitness(path, instance):
    distances = instance.n_shortest_edges_of_instance(8)
    print(distances)
    print(distances[0])
    temp = distances[0]

    print(temp)
    # for i in range(len(path)):
    #     if(distances)


def tsp_solver():
    tsp_size = 8
    num_libraries = tsp_size/2
    library_size = num_libraries/2

    num_libraries = int(num_libraries)
    library_size = int(library_size)

    tsp = tsp_generator(tsp_size)
    gene_libraries = gen_libraries(tsp, num_libraries, library_size)
    chromossome = gen_chromossome(gene_libraries, num_libraries)
    path = repair_chromossome(chromossome, tsp, num_libraries)

    fitness(path, tsp)

    return(path)

tsp_solver()