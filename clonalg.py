import numpy as np
from numpy.random import uniform

def affinity(cell, antigen):
    diff = np.ravel(cell - antigen)
    norm = np.sqrt(np.inner(diff, diff))
    return abs(1 - norm)



def create_random_cells(population_size, feature_num, feature_min, feature_max):
    return [
        uniform(low=0, high=1, size=feature_num)
        for _ in range(population_size)
    ]


def clone(cell, clone_rate):
    clone_num = int(clone_rate * cell[1])
    return [(cell[0], cell[1]) for _ in range(clone_num)]

def hypermutate_variability(cell, mutation_rate, antigen):
    genes = np.array(cell[0])

    mutation_mask = np.random.uniform(0, 1, size=genes.shape) < mutation_rate
    mutation_values = np.clip(np.random.uniform(-1, 1, size=genes.shape) * 
                              (1 - cell[1]) * 0.1 * mutation_mask, -0.3, 0.3)

    mutated_genes = np.clip(genes + mutation_values, 0, 1)

    return mutated_genes, affinity(mutated_genes, antigen)


def select(pop, pop_clones, pop_size):
    population = pop + pop_clones
    population = sorted(population, key=lambda x: x[1], reverse=True)[:pop_size]
    return population


def replace(population, population_rand, population_size):
    population = population + population_rand
    population = sorted(population, key=lambda x: abs(x[1]))[:population_size]

    return population

def remove_similar_clones(clone_population, sigma1):
    remaining_clones = []
    n_clones = len(clone_population)
    
    for i in range(n_clones):
        is_similar = False
        for j in range(i + 1, n_clones):
            if i != j:
                similarity = affinity(clone_population[i][0], clone_population[j][0])
                if similarity > sigma1:
                    is_similar = True
                    break
        if not is_similar:  # Aqui nós movemos a adição para fora do segundo loop
            remaining_clones.append(clone_population[i])
    return remaining_clones
