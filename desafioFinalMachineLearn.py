import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

df = pd.read_csv(r'C:\Desenvolvimento\Python\Python_CITHA\desafio\MachineLearn\produtividadeGuarana.csv', delimiter=';', decimal=',', encoding='UTF-8')

print("\n ",df.columns.tolist(), '\n')
df.rename(columns={ 
    'chuva_durante_floração_mm': 'chuva_flor', 
    'chuva_durante_colheita_mm': 'chuva_colheita', 
    'chuva_total_anual_mm': 'chuva_total', 
    'anomalia_chuva_floração_mm': 'anomalia_flor', 
    'temperatura_média_floração_C': 'temp_flor', 
    'umidade _relativa_média_floração_%': 'umid_flor', 
    'evento_ENSO': 'ENSO', 
    'produtividade_kg_por_ha': 'produtividade', 
    'produti vidade_safra ': 'safra' 
}, inplace=True)

# Conversão de escala fracionaria 80% -> 0.80
df['umid_flor'] = df['umid_flor'] / 100 
df.set_index('ano', inplace=True)

# Ver informações gerais do dataframe
df.info()

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Resumo estatístico
df.describe().T

# útil para visualizar tendências associadas a cada evento
def plot_boxplot_produtividade_por_ENSO():
    sns.set(style="whitegrid", palette="colorblind")
    sns.boxplot(
        data=df,
        x='ENSO',
        y='produtividade',
        order=['La Niña', 'Neutro', 'El Niño']
    )
    plt.title('Produtividade vs. Evento ENSO', fontsize=14)
    plt.xlabel('Evento ENSO', fontsize=12)
    plt.ylabel('Produtividade (kg/ha)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# relação entre a temperatura média durante a floração e a produtividade da safra
def plot_temp_flor_vs_produtividade():
    sns.scatterplot(data=df, x='temp_flor', y='produtividade', hue='ENSO', s=80, alpha=0.8)
    plt.title('Temperatura durante floração vs. Produtividade', fontsize=14)
    plt.xlabel('Temperatura média durante floração (°C)', fontsize=12)
    plt.ylabel('Produtividade (kg/ha)', fontsize=12)
    plt.legend(title='Evento ENSO')
    plt.tight_layout()
    plt.show()

# Essas observações são fundamentais para entender a distribuição dos dados e podem ajudar a identificar tendências ou problemas no dataset
def plot_histogramas_variaveis_numericas():
    df.select_dtypes(include='number').hist(bins=15, figsize=(12,8))
    plt.suptitle("Distribuições das Variáveis Numéricas")
    plt.tight_layout()
    plt.show()

# Essa análise é essencial para selecionar as variáveis mais relevantes para a modelagem e evitar problemas como a multicolinearidade, 
# que pode afetar a precisão dos modelos
def plot_matriz_correlacao():
    correlacao = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        correlacao,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": .8},
        vmin=-1, vmax=1
    )
    plt.title('Matriz de Correlação entre Variáveis Numéricas')
    plt.tight_layout()
    plt.show()

# É útil como uma etapa preliminar para avaliar a relação entre as variáveis antes de aplicar técnicas mais avançadas, como PCA (Análise de ComponentesPrincipais)
#  ou regressão
def plot_pairplot_variaveis_climaticas():
    colunas = ['chuva_flor', 'chuva_colheita', 'chuva_total', 
            'anomalia_flor', 'temp_flor', 'umid_flor', 'produtividade']
    sns.pairplot(
        df[colunas],
        corner=True,
        diag_kind='hist',
        plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'k'}
    )
    plt.suptitle("Matriz de Dispersão entre Variáveis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def criar_variaveis_derivadas():
# 1. Chuva relativa durante floração
    df['chuva_relativa'] = df['chuva_flor'] / df['chuva_total']
# 2. Binário: anomalia positiva ou não
    df['anomalia_bin'] = (df['anomalia_flor'] > 0).astype(int)

def aplicar_one_hot_encoding_ENSO():
    global df 
# 3. Codificar ENSO como variáveis dummies
    df = pd.get_dummies(df, columns=['ENSO'], drop_first=True) # cria ENSO_El Niño e ENSO_La Niña
    df.filter(like='ENSO').tail(10)
    print(df.head(10))

# Não comente ou remova essas duas funções, elas são de extrema importância para o tratamento de dados.
criar_variaveis_derivadas()
aplicar_one_hot_encoding_ENSO()

# 1. Definindo X e y
X = df.drop(columns=['produtividade', 'safra']) # safra é para classificação
y = df['produtividade'] 

# Lista de colunas numéricas
colunas_numericas = ['chuva_flor', 'chuva_colheita', 'chuva_total',
'anomalia_flor', 'temp_flor', 'umid_flor', 'chuva_relativa'] 
# Lista de colunas binárias
colunas_binarias = ['anomalia_bin', 'ENSO_La Niña', 'ENSO_Neutro'] 

# 3. Criando o transformador
preprocessador = ColumnTransformer(transformers=[('num', StandardScaler(),
colunas_numericas),('bin', 'passthrough', colunas_binarias)])

# shuffle=False = significa que os dados não foram embaralhados
# 4. Separando treino e teste sem embaralhar (respeitando ordem temporal)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
shuffle=False)

# Aplica o ColumnTransformer (padronização)
X_padronizado = preprocessador.fit_transform(X)
# Análise de Componentes Principais (PCA) = e transforma o conjunto original de variáveis em novas direções, chamadas componentes
def avaliação_variância_explicada():
    # Aplica PCA com todos os componentes (não limita n_components ainda)
    pca_full = PCA() 
    pca_full.fit(X_padronizado)
    # Scree Plot
    plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
    pca_full.explained_variance_ratio_, marker='o')
    plt.title('Scree Plot - Variância Explicada por Componente')
    plt.xlabel('Componente Principal')
    plt.ylabel('Proporção da Variância')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Mostrar numericamente
    for i, v in enumerate(pca_full.explained_variance_ratio_):
        print(f"PC{i+1}: {v:.2%}")



# Aplica PCA com 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_padronizado)

# Cria df_PCA com componentes e variáveis-alvo
df_PCA = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=X.index)
df_PCA['produtividade'] = df['produtividade']
df_PCA['safra'] = df['safra'] 
def aplicação_PCA_2cmpnt(df_PCA):
    sns.scatterplot(data=df_PCA, x='PC1', y='PC2', hue='safra', s=80, alpha=0.8)
    plt.title('PCA - Componentes Principais coloridos por Safra')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Safra')
    plt.tight_layout()
    plt.show()
  
# avaliação_variância_explicada()
# aplicação_PCA_2cmpnt()

# Regressão Linear Simples

# Pipeline: pré-processador + modelo
pipeline_original = make_pipeline(preprocessador, LinearRegression())

# Treinamento
pipeline_original.fit(X_train, y_train)

# Previsão
y_pred_orig = pipeline_original.predict(X_test)

# Avaliação
mse_orig = mean_squared_error(y_test, y_pred_orig)
rmse_orig = mse_orig ** 0.5
r2_orig = r2_score(y_test, y_pred_orig)
print(f"\n[Regressão linear] RMSE: {rmse_orig:.2f} | R²: {r2_orig:.2%}\n") 

# Regressão Linear com Regularização (Ridge)

# Pipeline com regularização L2 (Ridge)
lambda_regressao = 1 # testar vários valores para lambda
pipeline_ridge = make_pipeline(preprocessador, Ridge(alpha=lambda_regressao))

# Treinamento
pipeline_ridge.fit(X_train, y_train)

# Previsão
y_pred_ridge = pipeline_ridge.predict(X_test)

# Avaliação
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

rmse_ridge = mse_ridge ** 0.5
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"\n[Regularização Ridge (L²) | λ = {lambda_regressao}] RMSE:{rmse_ridge:.2f} | R²: {r2_ridge:.2%}\n")

# Regressão Linear Simples sobre PCA
# Definindo X e y com base no df_PCA
X_pca = df_PCA[['PC1', 'PC2']]
y_pca = df_PCA['produtividade']
# Divisão temporal (como fizemos antes)
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca,
y_pca, test_size=0.2, shuffle=False)
# Modelo linear simples com PCA
modelo_pca = LinearRegression()
modelo_pca.fit(X_pca_train, y_pca_train)
# Previsão
y_pred_pca = modelo_pca.predict(X_pca_test)
# Avaliação
rmse_pca = mean_squared_error(y_pca_test, y_pred_pca) ** 0.5
r2_pca = r2_score(y_pca_test, y_pred_pca)
print(f"[PCA + Regressão linear] RMSE: {rmse_pca:.2f} | R²: {r2_pca:.2%}")

# Regressão Linear com Regularização (Ridge) sobre PCA

# Modelo com Ridge sobre PCA
lambda_regressao = 1 # testar vários valores para lambda
modelo_pca_ridge = Ridge(alpha=lambda_regressao)
modelo_pca_ridge.fit(X_pca_train, y_pca_train)

# Previsão
y_pred_pca_ridge = modelo_pca_ridge.predict(X_pca_test)

# Avaliação
rmse_pca_ridge = mean_squared_error(y_pca_test, y_pred_pca_ridge) ** 0.5
r2_pca_ridge = r2_score(y_pca_test, y_pred_pca_ridge)
print(f"[PCA + Regularização Ridge (L²) | λ = {lambda_regressao}] RMSE:{rmse_pca_ridge:.2f} | R²: {r2_pca_ridge:.2%}")

# Simulação dos dados para plotagem
lambdas = [0.1, 1, 10, 100, 1000, 10000, 30000, 100000, 300000, 1000000]
rmse_sem_pca = [71.19, 68.30, 54.97, 34.91, 26.38, 25.61, 25.56, 25.55,
25.54, 25.54]
rmse_com_pca = [43.25, 42.98, 40.61, 31.20, 25.97, 25.57, 25.55, 25.54,
25.54, 25.54]

# R² para cada lambda (sem e com PCA)
r2_sem_pca = [-8.48, -7.72, -4.65, -1.28, -0.30, -0.23, -0.2221, -0.2206,
-0.2201, -0.2200]
r2_com_pca = [-2.50, -2.45, -2.08, -0.82, -0.2616, -0.2230, -0.2209, -0.2202,
-0.2200, -0.2199]

def simulacao_Plotagem():
    plt.plot(lambdas, rmse_sem_pca, marker='o', label='Sem PCA')
    plt.plot(lambdas, rmse_com_pca, marker='s', label='Com PCA')
    plt.xscale('log')
    plt.xlabel("λ (log scale)")
    plt.ylabel("RMSE")
    plt.title("Comparação do RMSE em função de λ (Ridge)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show() 
simulacao_Plotagem()

def simulacao_Plotagem2():
    plt.plot(lambdas, r2_sem_pca, marker='o', label='Sem PCA')
    plt.plot(lambdas, r2_com_pca, marker='s', label='Com PCA')
    plt.xscale('log')
    plt.xlabel("λ (log scale)")
    plt.ylabel("R²")
    plt.title("Comparação do R² em função de λ (Ridge)")
    plt.grid(True)
    32
    plt.legend()
    plt.tight_layout()
    plt.show()

# Relação entre Variáveis e Produtividade
def plot_modelos_para_variavel(x_var, X, y, scaler, pca_model, modelo_linear,modelo_ridge, modelo_pca, modelo_pca_ridge):
    x_index = X.columns.get_loc(x_var)
    x_vals = np.linspace(X[x_var].min(), X[x_var].max(), 100)
    X_mean = X.mean().to_numpy()
    X_input = np.tile(X_mean, (100, 1))
    X_input[:, x_index] = x_vals
    X_input_df = pd.DataFrame(X_input, columns=X.columns) # ⬅ usa os mesmos nomes
    X_input_scaled = scaler.transform(X_input_df)
    X_input_pca = pca_model.transform(X_input_scaled)
    y_linear = modelo_linear.predict(X_input_scaled)
    y_ridge = modelo_ridge.predict(X_input_scaled)
    y_pca = modelo_pca.predict(X_input_pca)
    y_pca_ridge = modelo_pca_ridge.predict(X_input_pca)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[x_var], y=y, color='red', label='Dados reais', s=50,edgecolor='black')
    plt.plot(x_vals, y_linear, label='Linear', linestyle='-', color='blue')
    plt.plot(x_vals, y_ridge, label='Ridge (λ=1.000.000)', linestyle='--',color='orange')
    plt.plot(x_vals, y_pca, label='PCA + Linear', linestyle='-.',color='green')
    plt.plot(x_vals, y_pca_ridge, label='PCA + Ridge (λ=100.000)',linestyle=':', color='purple')
    plt.xlabel(x_var)
    plt.ylabel('Produtividade (kg/ha)')
    plt.title(f'Comparação de modelos — {x_var}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 1. Reconstrução de X e y
X = df[[ 'chuva_flor', 'chuva_colheita', 'chuva_total', 'anomalia_flor', 'temp_flor', 'umid_flor', 'chuva_relativa' ]]
y = df['produtividade']

# 2. Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca_model = PCA(n_components=2)
X_pca = pca_model.fit_transform(X_scaled)

# Converte o array PCA em DataFrame com nomes
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index)
df[['PC1', 'PC2']] = df_pca

# 4. Modelos treinados separadamente
modelo_linear = LinearRegression().fit(X_scaled, y)
modelo_ridge = Ridge(alpha=1e6).fit(X_scaled, y)
modelo_pca = LinearRegression().fit(X_pca, y)
modelo_pca_ridge = Ridge(alpha=1e5).fit(X_pca, y)

plot_modelos_para_variavel('temp_flor', X, y, scaler, pca_model,modelo_linear, modelo_ridge, modelo_pca, modelo_pca_ridge)

#### Curva 1D da função custo
def plot_funcao_custo_1D(x_var, X, y, intervalo=(-200, 200), pontos=200):

# Plota a função de custo J(θ₁) para uma regressão univariada com a variável x_var.

    x = X[x_var].values
    y = y.values
    m = len(y)
    # Centraliza x para eliminar o intercepto implicitamente
    x_centralizado = x - x.mean()
    theta1_vals = np.linspace(intervalo[0], intervalo[1], pontos)
    custos = [(1 / (2 * m)) * np.sum((theta1 * x_centralizado - y) ** 2) for
    theta1 in theta1_vals]
    plt.figure(figsize=(8, 5))
    plt.plot(theta1_vals, custos)
    plt.xlabel("θ₁")
    plt.ylabel("J(θ₁)")
    plt.title(f"Função de Custo - {x_var} (x centralizado)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_funcao_custo_1D('temp_flor', X, y)

def plot_funcao_custo_2D(x_vars, X, y, range_theta=(-200, 200), pontos=100):
    """
    Plota a superfície da função de custo J(θ₁, θ₂) para duas variáveis.
    """
    x1 = X[x_vars[0]].values
    x2 = X[x_vars[1]].values
    y = y.values
    m = len(y)
    # Matriz de entrada com intercepto
    X_mat = np.vstack([np.ones(m), x1, x2]).T
    # Geração de grid de θ₁ e θ₂ (intercepto θ₀ fixado em 0 para simplificação)
    theta1_vals = np.linspace(range_theta[0], range_theta[1], pontos)
    theta2_vals = np.linspace(range_theta[0], range_theta[1], pontos)
    J_vals = np.zeros((pontos, pontos))
    for i in range(pontos):
        for j in range(pontos):
            theta = np.array([0, theta1_vals[i], theta2_vals[j]]) # θ₀ = 0
            h = X_mat @ theta
            J_vals[j, i] = (1 / (2 * m)) * np.sum((h - y) ** 2)

    # Superfície
    T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T1, T2, J_vals, cmap='viridis', edgecolor='none',
    alpha=0.9)
    ax.set_xlabel(f"θ₁ ({x_vars[0]})")
    38
    ax.set_ylabel(f"θ₂ ({x_vars[1]})")
    ax.set_zlabel("J(θ)")
    ax.set_title(f"Superfície da Função de Custo — {x_vars[0]} e {x_vars[1]}")
    plt.tight_layout()
    fig.subplots_adjust(right=0.5)
    plt.show()

    # Para um gráfico específico
plot_funcao_custo_2D(['temp_flor', 'chuva_flor'], X, y) 

"""#### Superfície 2D da função custo com PCA"""
def plot_funcao_custo_2D_PCA(X_pca, y, range_theta=(-200, 200), pontos=100):
    """
    Plota a superfície da função de custo J(θ₁, θ₂) usando os componentes
    principais PC1 e PC2.
    39
    """
    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1]
    m = len(y)
    # Matriz de entrada com intercepto
    X_mat = np.vstack([np.ones(m), pc1, pc2]).T
    # Grid de valores de θ₁ e θ₂
    theta1_vals = np.linspace(range_theta[0], range_theta[1], pontos)
    theta2_vals = np.linspace(range_theta[0], range_theta[1], pontos)
    J_vals = np.zeros((pontos, pontos))
    for i in range(pontos):
        for j in range(pontos):
            theta = np.array([0, theta1_vals[i], theta2_vals[j]])
            h = X_mat @ theta
            J_vals[j, i] = (1 / (2 * m)) * np.sum((h - y) ** 2)

    # Superfície 3D
    T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T1, T2, J_vals, cmap='viridis', edgecolor='none',
    alpha=0.9)
    ax.set_xlabel("θ₁ (PC1)")
    ax.set_ylabel("θ₂ (PC2)")
    ax.set_zlabel("J(θ)")
    ax.set_title("Superfície da Função de Custo — Componentes Principais(PCA)")
    fig.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.show()

plot_funcao_custo_2D_PCA(X_pca, y)

def plot_residuos(y_true, y_pred, titulo):
    """
    Plota os resíduos de um modelo específico.
    """
    residuos = y_true - y_pred
    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, residuos, color='royalblue', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Previsão")
    plt.ylabel("Resíduo")
    plt.title(f"Resíduos — {titulo}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\n")

    # Previsões
y_pred_linear = modelo_linear.predict(X_scaled)
y_pred_ridge = modelo_ridge.predict(X_scaled)
y_pred_pca = modelo_pca.predict(X_pca)
y_pred_pca_ridge = modelo_pca_ridge.predict(X_pca)

    # Gráficos de resíduos
plot_residuos(y, y_pred_linear, "Regressão Linear")
plot_residuos(y, y_pred_ridge, "Regressão Regularizada (λ = 1.000.000)")
plot_residuos(y, y_pred_pca, "PCA + Regressão Linear")
plot_residuos(y, y_pred_pca_ridge, "PCA + Regularizada (λ = 100.000)")

# 1. Mapeia a variável alvo (safra)
mapa_safra = {'baixa': 0, 'media': 1, 'alta': 2}
df['safra_num'] = df['safra'].map(mapa_safra)
# 2. Define variáveis preditoras (mesmas da regressão)
X_class = df[['chuva_flor', 'chuva_colheita', 'chuva_total', 'anomalia_flor',
'temp_flor', 'umid_flor', 'chuva_relativa']]
y_class = df['safra_num']
# 3. Padronização
scaler_class = StandardScaler()
46
X_class_scaled = scaler_class.fit_transform(X_class)
# 4. PCA (opcional — será usado para um dos modelos)
pca_class = PCA(n_components=2)
X_class_pca = pca_class.fit_transform(X_class_scaled)
# 5. Divisão treino/teste
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class_scaled, y_class, test_size=0.3, random_state=42, stratify=y_class)
X_train_pca, X_test_pca, _, _ = train_test_split(X_class_pca, y_class, test_size=0.3, random_state=42, stratify=y_class) 