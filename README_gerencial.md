# Relatório Gerencial — Otimização de Rede Neural para Detecção de Fraudes

Este documento apresenta um resumo gerencial do notebook `Ponderada_Yan_M_Coutinho_Semana_04.ipynb`, detalhando as etapas, decisões técnicas e resultados obtidos no projeto de otimização de um modelo de rede neural para detecção de fraudes em cartões de crédito.

## Objetivo
O objetivo principal foi aprimorar o desempenho de um modelo de rede neural pré-treinado, utilizando técnicas avançadas de ajuste de hiperparâmetros (grid search e random search), com foco em métricas como precisão, recall, F1-score e AUC-ROC. Também foi realizada uma comparação entre o modelo original e o modelo otimizado.

## Principais Etapas Realizadas

### 1. Análise Exploratória dos Dados
O trabalho iniciou-se com uma análise exploratória dos dados, onde foram visualizadas as primeiras linhas do dataset e suas estatísticas descritivas:

```python
df.head()
df.describe()
```

Foi possível observar que as variáveis principais (V1 a V28) já estavam normalizadas, mas as colunas `Time` e `Amount` apresentavam escalas diferentes. A análise dos histogramas revelou distribuições próximas da normal para as variáveis transformadas, mas com forte assimetria e presença de outliers em `Amount` e `Time`.

O balanceamento da variável alvo foi avaliado com:

```python
class_distribution = df['Class'].value_counts(normalize=True) * 100
print("Distribuição da coluna 'Class' em porcentagem:")
class_distribution
```

O resultado mostrou um forte desbalanceamento: menos de 0,2% das transações eram fraudes, evidenciando a necessidade de técnicas específicas para lidar com esse cenário, como usar class_weights no modelo.

### 2. Pré-processamento
Após identificar a necessidade de normalização das variáveis `Time` e `Amount`, o pré-processamento foi realizado com a separação dos dados em treino, validação e teste, utilizando amostragem estratificada:

```python
from sklearn.model_selection import train_test_split
X = df.drop('Class', axis=1)
y = df['Class']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10, random_state=42, stratify=y_train_val)
```

Os conjuntos ficaram bem distribuídos, mantendo a proporção de fraudes em todos os splits. A normalização foi aplicada apenas com os parâmetros do treino, evitando vazamento de informação:

```python
def normalize_and_save_params(df, columns_to_normalize, output_json_path='normalization_params.json'):
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    # ...salva parâmetros...
    return df_normalized

def apply_normalization(df, json_params_path='normalization_params.json'):
    # ...aplica normalização usando parâmetros salvos...
    return df_normalized

X_train_norm = normalize_and_save_params(X_train, ["Time", "Amount"])
X_val_norm = apply_normalization(X_val, 'normalization_params.json')
X_test_norm = apply_normalization(X_test, 'normalization_params.json')
```

A verificação dos histogramas após a normalização confirmou que as variáveis `Time` e `Amount` passaram a ter média zero e desvio padrão um no conjunto de treino.

### 3. Construção e Avaliação de Modelos
Na etapa de modelagem, foi definida uma arquitetura de rede neural densa, conforme o exemplo abaixo:

```python
def generate_model(optimizer='adam', loss='binary_crossentropy'):
    inputs = layers.Input(shape=(X_train_norm.shape[1],))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer, loss=loss, metrics=[*get_metrics()])
    return model

model = generate_model()
model.summary()
```

O modelo baseline, treinado com `binary_crossentropy` e otimizador Adam, apresentou bom desempenho em acurácia, mas baixo F1-score e recall para a classe de fraude, devido ao desbalanceamento. As métricas customizadas implementadas permitiram avaliar melhor o comportamento do modelo:

```python
def get_metrics():
    precision = Precision(name="precision")
    recall = Recall(name="recall")
    auc_roc = AUC(name='auc_roc')
    auc_pr = AUC(curve='PR', name='auc_pr')
    return [precision, recall, auc_roc, auc_pr]
```

Funções de perda alternativas foram testadas para melhorar a detecção de fraudes:

```python
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)
        return K.mean(weight * cross_entropy)
    return loss

def soft_f1_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    soft_f1 = 2 * tp / (2 * tp + fp + fn + K.epsilon())
    return 1 - soft_f1

def logit_adjusted_bce(pos_prior=0.01):
    adj = tf.cast(tf.math.log(pos_prior / (1 - pos_prior)), tf.float32)
    def loss(y_true, y_pred_logits):
        y_pred_logits = y_pred_logits - adj
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits))
    return loss
```

Com essas funções, o modelo passou a apresentar recall e F1-score significativamente melhores para a classe de fraude, mesmo que a acurácia global não tenha mudado muito.

### 4. Otimização de Hiperparâmetros
Para a otimização de hiperparâmetros, utilizou-se Random Search e Grid Search, conforme o exemplo:

```python
tuner_random = kt.RandomSearch(
    build_model,
    objective="val_precision",
    max_trials=5,
    overwrite=True,
    directory="my_dir",
    project_name="optimize_optimizer"
)
tuner_random.search(
    X_train_norm, y_train,
    epochs=10,
    validation_data=(X_val_norm, y_val),
    callbacks=[lr_scheduler]
)
```

A busca por hiperparâmetros resultou em uma configuração ótima que elevou o F1-score de aproximadamente 0.6 (baseline) para cerca de 0.8, além de ganhos em recall e AUC-ROC para a classe de fraude. O modelo final mostrou-se mais robusto e menos sensível ao desbalanceamento.

### 5. Avaliação dos Resultados
O desempenho dos modelos foi avaliado por meio de gráficos gerados pela função:

```python
def plot_history_data(history):
    # ...plota loss, precision, recall, F1-score, AUC-ROC, AUC-PR...
```

Os gráficos de aprendizado evidenciaram a evolução das métricas ao longo dos epochs, mostrando que o modelo otimizado não apenas melhorou o F1-score, mas também manteve estabilidade entre treino e validação, indicando menor risco de overfitting.

### 6. Discussão e Limitações
Por fim, a conclusão do trabalho destacou a melhora do F1-Score de 0.6 para 0.8 após a otimização dos hiperparâmetros. No entanto, observou-se que o modelo ainda apresenta dificuldades de generalização devido ao ruído causado por outliers presentes no conjunto de dados. Recomenda-se a aplicação de técnicas de detecção e remoção de outliers para aprimorar ainda mais a robustez e a capacidade de generalização do modelo em cenários reais.
