import numpy as np
import clonalg

def train_ais_classifier(train, feature_num, feature_min, feature_max, population_size=500, selection_size=70, memory_set_percentage=20, clone_rate=30, mutation_rate=0.2, stop_condition=50, d=15, sigma1=0.4, sigma2=0.4):
    stop = 0
    population = clonalg.create_random_cells(population_size, feature_num, feature_min, feature_max)

    while stop != stop_condition:
        for antigen in train:
            population_affinity = [(cell, clonalg.affinity(cell, antigen)) for cell in population]
            population_affinity = sorted(population_affinity, key=lambda x: abs(x[1]))
            best_affinity = population_affinity[:selection_size]

            clone_population = []
            for cell in best_affinity:
                cell_clones = clonalg.clone(cell, clone_rate)
                clone_population += cell_clones

            mutated_clone_population = []
            for cell in clone_population:
                mutated_clone = clonalg.hypermutate_variability(cell, mutation_rate, antigen)
                mutated_clone_population.append(mutated_clone)

            mutated_clone_population.sort(key=lambda x: x[1])
            pop_size = round(len(clone_population) / 100) * memory_set_percentage
            mutated_clone_population = mutated_clone_population[:pop_size]

            filtered_clone_population = list(filter(lambda x: x[1] > sigma2, mutated_clone_population))
            remaining_clone_population = clonalg.remove_similar_clones(filtered_clone_population, sigma1)

            remaining_clone_population_no_affinity = [(cell[0],) for cell in remaining_clone_population]
            population += remaining_clone_population_no_affinity

        population = clonalg.suppress_similar_cells(population, sigma1)

        new_cells = clonalg.create_random_cells(int(population_size * (d / 100)), feature_num, feature_min, feature_max)
        population += new_cells
        print("População: ", len(population), "     Iteração: ", stop)
        stop += 1

    for i, cell in enumerate(population):
        if len(cell[0].shape) > 1:
            flattened_cell = np.ravel(cell[0])
            population[i] = (flattened_cell,)
        else:
            population[i] = cell

        if isinstance(cell, tuple):
            array_cell = np.array(cell)
            flattened_cell = np.ravel(array_cell)
            population[i] = (flattened_cell)

    return population

import numpy as np

def multiclass_performance_measure(populations, test_data):
    correct_classifications = 0
    total_samples = test_data.shape[0]

    for row in test_data:
        sample_features = row[:-1]
        true_label = int(row[-1])

        class_scores = []
        for population in populations:
            if population:
                class_score = sum(clonalg.affinity(cell, sample_features) for cell in population)
                class_scores.append(class_score)
            else:
                class_scores.append(0)

        predicted_label = class_scores.index(max(class_scores))

        if predicted_label == true_label:
            correct_classifications += 1

    return correct_classifications / total_samples

import concurrent.futures

def train_clonalg_parallel(train_data, params, classes):
    def train_clonalg_single_class(train_data, params, target_class):
        # Filtrar dados de treino para a classe alvo
        train_subset = train_data.loc[train_data[train_data.columns[-1]] == target_class].drop(columns=[train_data.columns[-1]])
        # Treinar a população usando os dados filtrados
        train_subset_array = train_subset.values
        population = train_ais_classifier(train_subset_array, **params)
        return population
    
    trained_populations = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_clonalg_single_class, train_data, params, class_label) for class_label in classes]
        for future in concurrent.futures.as_completed(futures):
            trained_populations.append(future.result())

    return trained_populations

def multiclass_performance_measure_tolerance(trained_populations, test_data, tolerance):
    correct_classifications = 0
    total_samples = test_data.shape[0]

    for _, row in test_data.iterrows():
        sample_features = row[:-1].values
        true_label = int(row[-1])

        class_scores = []
        for population in trained_populations:
            class_score = sum(affinity(cell, sample_features) for cell in population)
            class_scores.append(class_score)

        predicted_label = class_scores.index(min(class_scores))
        min_distance = min(class_scores)

        if predicted_label == true_label and min_distance <= tolerance:
            correct_classifications += 1

    return correct_classifications / total_samples

