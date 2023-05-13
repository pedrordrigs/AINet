from train_loop import train_clonalg_parallel
from train_loop import multiclass_performance_measure_v2
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('dataset/iris.csv', header=None)

df[4] = pd.Categorical(df[4]).codes
labels = df[[4]].copy()

df = df.drop(columns=[4])

# Standard Scaler
scaler = preprocessing.MinMaxScaler()
train = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
train = train.join(labels)
# Dividir o conjunto de dados em treino e teste
train_data, test_data = train_test_split(train, test_size=0.3, random_state=82)

params = {
    'population_size': 300,
    'selection_size': 20,
    'memory_set_percentage': 15,
    'clone_rate': 10,
    'mutation_rate': 0.3,
    'stop_condition': 10,
    'd': 5,
    'sigma1': 0.98,
    'sigma2': 0.85,
    'feature_num': train_data.shape[1] - 1,
    'feature_min': train_data.min().min(),
    'feature_max': train_data.max().max(),
}

trained_populations = train_clonalg_parallel(train_data, params, [0, 1, 2])

# Avaliar o desempenho do classificador AIS
test_data_array = test_data.values
performance, ltrue, lpredict = multiclass_performance_measure_v2(trained_populations, test_data_array)
print(f"Desempenho do classificador AIS: {performance * 100:.2f}%")
print(lpredict)
print(ltrue)