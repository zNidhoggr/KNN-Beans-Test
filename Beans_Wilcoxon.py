import numpy as np
import pandas as pd
import sklearn.preprocessing
from scipy.io import arff
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import wilcoxon

# Carregar a base de dados "Dry_Bean_Dataset.arff"
dados, metadata = arff.loadarff('D:/UFCA/Aprendizado de Maquina/DryBeanDataset/Dry_Bean_Dataset.arff')
df = pd.DataFrame(dados)

# Separar features (X) e o target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Normalizar as features
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Codificar o target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse=False)
y_encoded = y_encoded.reshape(-1, 1)
y_onehot = onehot_encoder.fit_transform(y_encoded)

X = np.array(X)
y = y_onehot

# Configurações de K e distâncias para testar
K_values = [5, 10, 15]
distances = ['euclidean', 'manhattan', 'chebyshev']

# Inicializar listas para armazenar os resultados
results = []
configurations = []

# Loop para experimentar cada configuração
for K in K_values:
    for distance in distances:
        knn = KNeighborsClassifier(n_neighbors=K, metric=distance)
        cv_scores = cross_val_score(knn, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
        results.append(cv_scores)
        configurations.append((K, distance))

# Comparação estatística usando o teste Wilcoxon
p_values = []
for i in range(len(results)):
    for j in range(i+1, len(results)):
        _, p_value = wilcoxon(results[i], results[j])
        p_values.append(p_value)

# Threshold de significância (nível de confiança)
alpha = 0.05

# Contar quantas configurações são estatisticamente diferentes de cada uma das outras
statistically_different = sum(1 for p_value in p_values if p_value < alpha)

# Mostrar os resultados
for i, config in enumerate(configurations):
    print(f"Configuração {config}: Média dos scores de validação cruzada = {np.mean(results[i]):.4f}")

print(f"\nNúmero de configurações estatisticamente diferentes de outras configurações: {statistically_different}")
