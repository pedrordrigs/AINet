import numpy as np
from numpy.random import uniform


def affinity(cell, antigen):
    aff = 0
    for i in range(len(cell)):
        aff += (cell[i] - antigen[i])
    return aff


def create_random_cells(population_size, feature_num, feature_min, feature_max):
    # Gerar individuos inicializados com valores aleatórios utilizando o argming e argmax de cada feature do dataset
    population = [uniform(low=feature_min, high=feature_max, size=feature_num) for x in range(population_size)]
    return population


def clone(p_i, clone_rate):
    clone_num = int(clone_rate / p_i[1])
    clones = [(p_i[0], p_i[1]) for x in range(clone_num)]

    return clones


def hypermutate(cell, mutation_rate, feature_min, feature_max, antigen):
    if uniform() <= cell[1] / (mutation_rate * 100):
        ind_tmp = []
        for gen in cell[0]:
            if uniform() <= cell[1] / (mutation_rate * 100):
                ind_tmp.append(uniform(low=feature_min, high=feature_max))
            else:
                ind_tmp.append(gen)

        return (np.array(ind_tmp), affinity(cell, antigen))
    else:
        return cell


def select(pop, pop_clones, pop_size):
    population = pop + pop_clones
    population = sorted(population, key=lambda x: x[1].any())[:pop_size]

    return population


def replace(population, population_rand, population_size):
    population = population + population_rand
    population = sorted(population, key=lambda x: x[1])[:population_size]

    return population
