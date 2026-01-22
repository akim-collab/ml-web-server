import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Генерация данных
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Создание DataFrame
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df['target'] = y

# Сохранение
df.to_csv('data/sample_data.csv', index=False)
print(f"Сгенерировано {len(df)} записей с {X.shape[1]} признаками")