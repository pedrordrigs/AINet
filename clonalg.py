import numpy as np
from numpy.random import uniform

def affinity(cell, antigen):
    aff = 0
    for i in range(len(cell)):
        aff += (cell[i] - antigen[i])
    return aff


def create_random_cells(population_size, feature_num, feature_min, feature_max):
    population = [uniform(low=feature_min, high=feature_max, size=feature_num) for x in range(population_size)]
    return population


def clone(cell, clone_rate):
    clone_num = int(clone_rate / cell[1])
    clones = [(cell[0], cell[1]) for x in range(clone_num)]

    return clones

def hypermutate_variability(cell, mutation_rate, antigen):
    genes = cell[0]
    clone_cache = []
    for gen in genes:
        clone_cache.append(gen + (uniform(0, 1) * (mutation_rate * cell[1])))

    return (np.array(clone_cache), abs(affinity(clone_cache, antigen)))

def hypermutate(cell, mutation_rate, antigen, feature_min, feature_max):
    if uniform() <= abs(cell[1]) / (mutation_rate * 100):
        ind_tmp = []
        for gen in cell[0]:
            if uniform() <= abs(cell[1]) / (mutation_rate * 100):
                ind_tmp.append(uniform(low=feature_min, high=feature_max))
            else:
                ind_tmp.append(gen)

        return (np.array(ind_tmp), abs(affinity(cell, antigen)))
    else:
        return cell


def select(pop, pop_clones, pop_size):
    population = pop + pop_clones
    population = sorted(population, key=lambda x: abs(x[1]).any())[:pop_size]

    return population


def replace(population, population_rand, population_size):
    population = population + population_rand
    population = sorted(population, key=lambda x: abs(x[1]))[:population_size]

    return population

