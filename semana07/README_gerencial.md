# Relatório Gerencial — Implementação de Rede Neural LSTM para Detecção de Fraudes

Este documento apresenta um resumo gerencial do notebook `Ponderada_ErikRafaYan.ipynb`, detalhando as etapas, decisões técnicas e resultados obtidos no projeto de implementação de uma rede neural LSTM (Long Short-Term Memory) para detecção de fraudes em transações financeiras utilizando o dataset IEEE-CIS Fraud Detection.

## Objetivo
O objetivo principal foi desenvolver um modelo de rede neural LSTM capaz de detectar fraudes em transações financeiras, aproveitando as características temporais dos dados através de sequências. O projeto focou na análise exploratória dos dados, preparação adequada para modelos sequenciais, definição da arquitetura LSTM, e avaliação do desempenho do modelo usando métricas apropriadas para dados desbalanceados.

## Principais Etapas Realizadas

### 1. Setup e Carregamento dos Dados
O trabalho iniciou-se com a instalação das dependências necessárias e o carregamento do dataset IEEE-CIS Fraud Detection através do Google Drive:

```python
%pip install gdown tensorflow tf-keras matplotlib keras-tuner gdown
doc_id = "1u_OWAPkIdgJw1ah5xP_dGBFMSANxjxEl"
URL = f"https://drive.google.com/uc?id={doc_id}"
gdown.download(URL, arquivo_destino_colab, quiet=False)
```

O dataset carregado possui 151MB e contém variáveis já transformadas (V1-V28), além das variáveis Time, Amount e a variável alvo Class.

### 2. Análise Exploratória dos Dados
Foi realizada uma análise exploratória abrangente dos dados, incluindo visualização das primeiras linhas e estatísticas descritivas:

```python
df.head()
df.describe()
```

#### Análise das Distribuições
Através de histogramas para todas as variáveis, foi observado que:
- As componentes V1-V28 já estavam normalizadas
- As variáveis Time e Amount necessitavam normalização devido à diferença de escala
- Não havia valores textuais que precisassem de codificação

#### Distribuição da Classe Alvo
A análise do balanceamento revelou forte desbalanceamento entre as classes, similar ao dataset da semana 04:

```python
class_distribution = df['Class'].value_counts(normalize=True) * 100
```

Este desbalanceamento destacou a necessidade de métricas específicas e técnicas de balanceamento de classes.

#### Análise de Correlação
Foi criado um mapa de calor das correlações, focando apenas em correlações absolutas maiores que 0.03:

```python
correlacao_base = df.corr()
correlacao = correlacao_base.where(correlacao_base.abs() >= 0.03)
sns.heatmap(correlacao, annot=True, cmap="coolwarm")
```

A análise revelou que variáveis como Time, V8, V13, V15, V20, V22, V23, V24, V25, V26, V27, V28 não apresentavam forte correlação com a variável dependente Class.

#### Análise de Outliers
Utilizando boxplots, foi identificada a presença significativa de outliers em todas as variáveis, indicando a necessidade de tratamento durante o pré-processamento.

### 3. Pré-processamento Específico para LSTM
O pré-processamento foi adaptado para as necessidades específicas de redes neurais recorrentes:

#### Separação Temporal dos Dados
Devido à natureza temporal do LSTM, os dados foram ordenados pela variável Time e separados sequencialmente:

```python
seq_length = 30
test_size = 0.15
val_size = 0.10

df_sorted = df.sort_values(by="Time").reset_index(drop=True)
X = df_sorted.drop(['Class', 'Time'], axis=1).values
y = df_sorted['Class'].values

n = len(df_sorted)
n_test = int(n * test_size)
n_val = int((n - n_test) * val_size)

X_train = X[:-(n_test + n_val)]
y_train = y[:-(n_test + n_val)]
```

Os conjuntos resultantes foram: Treino (217.878), Validação (24.208) e Teste (42.721).

#### Remoção de Outliers
Foi implementada uma função para remoção de outliers baseada no método IQR:

```python
def remove_outliers(df, factor=1.5):
    for col in df_clean.select_dtypes(include=np.number).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
```

#### Normalização
A normalização foi aplicada apenas à variável Amount, mantendo os parâmetros do conjunto de treino:

```python
X_train_norm = normalize_and_save_params(X_train_df, columns_to_normalize=['Amount'])
X_val_norm = apply_normalization(X_val_df, 'normalization_params.json')
X_test_norm = apply_normalization(X_test_df, 'normalization_params.json')
```

#### Geração de Sequências Temporais
Para adequar os dados ao modelo LSTM, foram criadas janelas temporais de 30 passos:

```python
train_gen = TimeseriesGenerator(X_train_norm, y_train, length=seq_length, batch_size=32)
val_gen = TimeseriesGenerator(X_val_norm, y_val, length=seq_length, batch_size=32)
test_gen = TimeseriesGenerator(X_test_norm, y_test, length=seq_length, batch_size=32)
```

O generator de treino resultou em 2.482 batches.

### 4. Construção e Arquitetura do Modelo LSTM
Foi definida uma arquitetura LSTM com múltiplas camadas:

```python
def generate_model(optimizer='adam', loss='binary_crossentropy'):
    model = Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=(seq_length, X_train_norm.shape[1])))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(32))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=[*get_metrics()])
    return model
```

A arquitetura incluiu:
- Primeira camada LSTM com 64 neurônios e return_sequences=True
- Primeira camada Dropout (0.2)
- Segunda camada LSTM com 32 neurônios
- Segunda camada Dropout (0.2)
- Camada densa final com ativação sigmoid

### 5. Métricas de Avaliação
Foram implementadas métricas específicas para dados desbalanceados:

```python
def get_metrics():
    precision = Precision(name="precision")
    recall = Recall(name="recall")
    auc_roc = AUC(name='auc_roc')
    auc_pr = AUC(curve='PR', name='auc_pr')
    return [precision, recall, auc_roc, auc_pr]
```

### 6. Treinamento do Modelo Baseline
O modelo foi treinado utilizando class weights para balanceamento:

```python
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

history = model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_callback]
)
```

### 7. Visualização e Análise dos Resultados
Foi implementada uma função para plotar múltiplas métricas:

```python
def plot_history_data(history):
    # Plot Loss, Precision, Recall, F1-Score, AUC-ROC, AUC-PR
```

A função gera gráficos para:
- Loss de treino e validação
- Precision de treino e validação
- Recall de treino e validação
- F1-Score calculado e plotado
- AUC-ROC de treino e validação
- AUC-PR de treino e validação

### 8. Resultados e Limitações
O modelo baseline apresentou problemas significativos de underfitting:
- Precision no treino: próxima de 0
- Recall no treino: próxima de 0
- AUC-ROC de validação: em torno de 0.52 (próximo ao aleatório)
- Loss de validação: alta (em torno de 0.08)

O underfitting observado indica que o modelo não conseguiu aprender adequadamente os padrões dos dados, sugerindo a necessidade de:
- Ajustes na arquitetura (mais neurônios, mais camadas)
- Modificações nos hiperparâmetros
- Alterações na função de perda
- Revisão do pré-processamento

### 9. Conclusões e Próximos Passos
O projeto implementou com sucesso uma arquitetura LSTM para detecção de fraudes, mas o modelo baseline mostrou performance insatisfatória. Os principais desafios identificados foram:

1. **Underfitting Severo**: O modelo não conseguiu aprender os padrões nos dados de treino
2. **Complexidade dos Dados**: O dataset IEEE-CIS apresenta características mais complexas que requerem arquiteturas mais sofisticadas
3. **Balanceamento de Classes**: Apesar do uso de class weights, o modelo ainda não conseguiu identificar adequadamente a classe minoritária

**Recomendações para melhoria**:
- Experimentar arquiteturas LSTM mais profundas
- Implementar técnicas de regularização avançadas
- Testar diferentes funções de perda (focal loss, por exemplo)
- Aplicar técnicas de aumento de dados (data augmentation)
- Considerar arquiteturas híbridas (CNN+LSTM)
- Otimizar hiperparâmetros usando Keras Tuner

O projeto estabeleceu uma base sólida para desenvolvimento futuro, com pipeline completo de pré-processamento temporal e avaliação adequada para dados desbalanceados.
