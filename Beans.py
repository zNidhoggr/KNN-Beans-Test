import numpy as np
import pandas as pd
import sklearn.preprocessing
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

dados, metadata = arff.loadarff('D:/UFCA/Aprendizado de Maquina/DryBeanDataset/Dry_Bean_Dataset.arff')
df = pd.DataFrame(dados)

X = df.drop('Class', axis=1)
y = df['Class']

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse=False)
y_encoded = y_encoded.reshape(-1, 1)
y_onehot = onehot_encoder.fit_transform(y_encoded)

X = np.array(X)
y = y_onehot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

testpatterns = X_test[:1000]
print("Test Patterns:")
print(testpatterns)

def apply_knn(nn, X_train, y_train, testpattern):
    nn.fit(X_train, y_train)
    res = nn.kneighbors(testpattern, return_distance=True)

    nn_distances = res[0]
    nn_indices = res[1]
    nn_classes = [label_encoder.inverse_transform([np.argmax(y_train[index])])[0] for index in nn_indices]

    return nn_classes, nn_distances

def print_results(nn_classes, nn_distances, k):
    for i, (predicted_class, distance) in enumerate(zip(nn_classes, nn_distances)):
        print(f'Vizinho {i+1}: Classe: {predicted_class}, Distância: {distance[0]}, usando {k} vizinhos')

print("\nNN1")
nn1 = KNeighborsClassifier(n_neighbors=5)
y_pred_nn1 = []
for testpattern in testpatterns:
    testpattern = testpattern.reshape(1, -1)
    nn_classes, nn_distances = apply_knn(nn1, X_train, y_train, testpattern)
    y_pred_nn1.append(nn_classes[0])
    print_results(nn_classes, nn_distances, nn1.n_neighbors)

y_pred_nn1 = np.array(y_pred_nn1)

print("\nNN2")
nn2 = KNeighborsClassifier(n_neighbors=10)
y_pred_nn2 = []
for testpattern in testpatterns:
    testpattern = testpattern.reshape(1, -1)
    nn_classes, nn_distances = apply_knn(nn2, X_train, y_train, testpattern)
    y_pred_nn2.append(nn_classes[0])
    print_results(nn_classes, nn_distances, nn2.n_neighbors)

y_pred_nn2 = np.array(y_pred_nn2)

print("\nNN3")
nn3 = KNeighborsClassifier(n_neighbors=15)
y_pred_nn3 = []
for testpattern in testpatterns:
    testpattern = testpattern.reshape(1, -1)
    nn_classes, nn_distances = apply_knn(nn3, X_train, y_train, testpattern)
    y_pred_nn3.append(nn_classes[0])
    print_results(nn_classes, nn_distances, nn3.n_neighbors)

y_pred_nn3 = np.array(y_pred_nn3)


y_pred_nn1_onehot = onehot_encoder.transform(label_encoder.transform(y_pred_nn1).reshape(-1, 1))
y_pred_nn2_onehot = onehot_encoder.transform(label_encoder.transform(y_pred_nn2).reshape(-1, 1))
y_pred_nn3_onehot = onehot_encoder.transform(label_encoder.transform(y_pred_nn3).reshape(-1, 1))

print("\nAcurácia do modelo NN1:", accuracy_score(y_test[:1000], y_pred_nn1_onehot))
print("Acurácia do modelo NN2:", accuracy_score(y_test[:1000], y_pred_nn2_onehot))
print("Acurácia do modelo NN3:", accuracy_score(y_test[:1000], y_pred_nn3_onehot))