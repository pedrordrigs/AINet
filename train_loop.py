import numpy as np
import clonalg

def train_ais_classifier(train, feature_num, feature_min, feature_max, population_size, selection_size, memory_set_percentage, clone_rate, mutation_rate, stop_condition, d, sigma1, sigma2):
    stop = 0
    population = clonalg.create_random_cells(population_size, feature_num, feature_min, feature_max)

    while stop != stop_condition:
        # 1. For each antigen do
        for antigen in train:
            # 1.1 Determine its affinity to network cells
            population_affinity = [(cell, clonalg.affinity(cell, antigen)) for cell in population]
            # 1.2 Select the n highest affinity network cells
            population_affinity = sorted(population_affinity, key=lambda x: abs(x[1]))
            best_affinity = population_affinity[selection_size:]
            # 1.3 Generate Nc clones from these n cells. The higher the affinity, the larger Nc;
            clone_population = []
            for cell in best_affinity:
                cell_clones = clonalg.clone(cell, clone_rate)
                clone_population += cell_clones
            # 1.4 Apply hypermutation to the generated clones, with variability inversely proportional to the progenitor fitness 
            # 1.5 Determine the affinity among the antigen and all clones
            mutaded_clone_population = []
            for cell in clone_population:
                mutated_clone = clonalg.hypermutate_variability(cell, mutation_rate, antigen)
                mutaded_clone_population.append(mutated_clone)
            # 1.6 Keep only m% of the highest affinity mutated clones into the clone population
            mutaded_clone_population.sort(key=lambda x: x[1])
            pop_size = round(len(clone_population)/100)*memory_set_percentage

            mutaded_clone_population = mutaded_clone_population[pop_size:]
            # 1.7 Eliminate all clones but one whose affinity with the antigen is inferior to a predefined threshold sigma2 (apoptosis)
            filtered_clone_population = list(filter(lambda x: x[1] > sigma2, mutaded_clone_population))
            # 1.8 Determine the affinity among all the mutated clonesand eliminate those whose affinity with each other is above a pre-defined threshold sigma1 (supression)
            remaining_clone_population = clonalg.remove_similar_clones(filtered_clone_population, sigma1)

            # 1.9 Insert the remaining clones into the populatuon
            # Remova o atributo de afinidade das células em remaining_clone_population
            remaining_clone_population_no_affinity = [(cell[0],) for cell in remaining_clone_population]
            # Adicione remaining_clone_population_no_affinity à população
            population = population + remaining_clone_population_no_affinity

        # 2.0 Determine the simillarity among all the antibodies and eliminate those with similarity above a threshold sigma1 (supression)
        population = clonalg.remove_similar_clones(population, sigma1)

    # 3 Introduce a d% of new randomly generated cells (random insertion)
        if(stop != stop_condition):
            new_cells = clonalg.create_random_cells(int(population_size * (d / 100)), feature_num, feature_min, feature_max)
            population += new_cells

        for i, cell in enumerate(population):
            # Verifica se a célula tem mais de uma dimensão
            if len(cell[0].shape) > 1:
                # Aplica numpy.ravel() para simplificar as dimensões
                flattened_cell = np.ravel(cell[0])
                population[i] = (flattened_cell,)
            else:
                population[i] = cell

            if isinstance(cell, tuple):
                array_cell = np.array(cell)
                flattened_cell = np.ravel(array_cell)
                population[i] = (flattened_cell)

        print("População: ", len(population), "     Iteração: ", stop)
        stop += 1
    return population


#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////UTILS/////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////

import concurrent.futures

def train_clonalg_single_class(train_data, params, target_class):
    # Filtrar dados de treino para a classe alvo
    train_subset = train_data.loc[train_data[train_data.columns[-1]] == target_class].drop(columns=[train_data.columns[-1]])
    # Treinar a população usando os dados filtrados
    train_subset_array = train_subset.values
    population = train_ais_classifier(train_subset_array, **params)
    return population

def train_clonalg_parallel(train_data, params, classes):
    trained_populations = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(train_clonalg_single_class, train_data, params, class_label) for class_label in classes]
        for future in concurrent.futures.as_completed(futures):
            trained_populations.append(future.result())
    return trained_populations


# def multiclass_performance_measure_v2(populations, test_data):
#     correct_classifications = 0
#     total_samples = test_data.shape[0]
#     true_labels = []
#     predicted_labels = []

#     for row in test_data:
#         sample_features = row[:-1]
#         true_label = int(row[-1])
#         true_labels.append(true_label)

#         class_scores = []
#         for population in populations:
#             if len(population) > 0:
#                 max_affinity = max(clonalg.affinity(cell, sample_features) for cell in population)
#                 class_scores.append(max_affinity)
#             else:
#                 class_scores.append(0)

#         predicted_label = class_scores.index(max(class_scores))
#         predicted_labels.append(predicted_label)

#         if predicted_label == true_label:
#             correct_classifications += 1

#     accuracy = correct_classifications / total_samples
#     return accuracy, true_labels, predicted_labels

def multiclass_performance_measure_v2(populations, test_data):
    k = 3
    correct_classifications = 0
    total_samples = test_data.shape[0]
    true_labels = []
    predicted_labels = []

    for row in test_data:
        sample_features = row[:-1]
        true_label = int(row[-1])
        true_labels.append(true_label)

        class_scores = []
        for population in populations:
            if len(population) > 0:
                # Calcule as k maiores afinidades para cada população
                k_largest_affinities = sorted([clonalg.affinity(cell, sample_features) for cell in population], reverse=True)[:k]
                # Some as k maiores afinidades para obter o escore da classe
                class_score = sum(k_largest_affinities)
                class_scores.append(class_score)
            else:
                class_scores.append(0)

        predicted_label = class_scores.index(max(class_scores))
        predicted_labels.append(predicted_label)

        if predicted_label == true_label:
            correct_classifications += 1

    accuracy = correct_classifications / total_samples
    return accuracy, true_labels, predicted_labels