import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from train_loop import multiclass_performance_measure_v2
from sklearn.metrics import confusion_matrix, classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import itertools
from sklearn.model_selection import train_test_split
from train_loop import train_clonalg_parallel

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

df = pd.read_csv('iris.csv', header=None)
df[4] = pd.Categorical(df[4]).codes
labels = df[[4]].copy()

df = df.drop(columns=[4])

# Standard Scaler
scaler = preprocessing.StandardScaler()
train = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
train = train.join(labels)

train_data, test_data = train_test_split(train, test_size=0.3, random_state=42)

# Definir os limites e intervalos de variação para os parâmetros
d_values = [13, 11, 9, 7, 5]
clone_rate_values = [40, 45, 50, 55]

# Realizar todas as combinações possíveis de d e clone_rate
for d, clone_rate in itertools.product(d_values, clone_rate_values):
    params = {
        'population_size': 400,
        'selection_size': 50,
        'memory_set_percentage': 50,
        'clone_rate': clone_rate,
        'mutation_rate': 0.3,
        'stop_condition': 100,
        'd': d,
        'sigma1': 0.3,
        'sigma2': 0.3,
        'feature_num': train_data.shape[1] - 1,
        'feature_min': train_data.min().min(),
        'feature_max': train_data.max().max(),
    }

    # Abra um arquivo para salvar a saída
    output_filename = f"output_d_{d}_clone_rate_{clone_rate}.txt"
    with open(output_filename, "w") as output_file:
        # Imprimir os parâmetros utilizados no início do arquivo
        print(f"Parâmetros utilizados:\n", file=output_file)
        print(f"d: {d}", file=output_file)
        print(f"clone_rate: {clone_rate}\n", file=output_file)

        # Crie um objeto KFold
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        # Armazenar as métricas de desempenho de cada iteração
        performance_scores = []
        all_true_labels = []
        all_predicted_labels = []

        # Loop através das divisões de treinamento e teste
        for i, (train_index, test_index) in enumerate(kfold.split(train), start=1):
            # Dividir o conjunto de dados em treinamento e teste com base nos índices
            train_data = train.iloc[train_index]
            test_data = train.iloc[test_index]

            # Treinar o classificador AIS
            trained_populations = train_clonalg_parallel(train_data, params, [0, 1, 2])

            # Avaliar o desempenho do classificador AIS
            test_data_array = test_data.values
            accuracy, true_labels, predicted_labels = multiclass_performance_measure_v2(trained_populations, test_data_array)
            performance_scores.append(accuracy)

            # Estender all_true_labels e all_predicted_labels com os rótulos verdadeiros e previstos da iteração atual
            all_true_labels.extend(true_labels)
            all_predicted_labels.extend(predicted_labels)

            # Imprimir o relatório de classificação para a iteração atual
            print("\n//////////////////////////////////////////////\n", file=output_file)
            print(f"Relatório de classificação para a pasta {i}:", file=output_file)
            print(classification_report(true_labels, predicted_labels), file=output_file)

        # Calcular a precisão média
        print("\n//////////////////////////////////////////////\n", file=output_file)
        mean_accuracy = np.mean(performance_scores)
        print(f"Desempenho médio do classificador AIS: {mean_accuracy * 100:.2f}%", file=output_file)

        # Calcular e exibir a matriz de confusão combinada
        print("\n//////////////////////////////////////////////\n", file=output_file)
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        print("Matriz de confusão:", file=output_file)
        print(conf_matrix, file=output_file)

        # Calcular e exibir o relatório de classificação combinado
        print("\n//////////////////////////////////////////////\n", file=output_file)
        class_report = classification_report(all_true_labels, all_predicted_labels)
        print("Relatório de classificação:", file=output_file)
        print(class_report, file=output_file)
