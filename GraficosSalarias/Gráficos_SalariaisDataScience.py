import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregar os dados do arquivo CSV
df = pd.read_csv("data_science_salaries.csv")

# Filtrar apenas os dados do Brasil
df_brazil = df[df['employee_residence'] == 'Brazil']

# Selecionar colunas relevantes e codificar categoricamente
df_brazil['experience_level_code'] = df_brazil['experience_level'].map({
    'Entry-level': 0, 'Mid-level': 1, 'Senior-level': 2, 'Executive-level': 3
})
df_brazil['job_title_code'] = df_brazil['job_title'].astype('category').cat.codes

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_brazil[['job_title_code', 'experience_level_code', 'salary_in_usd']])

# Aplicar PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# Criar o gráfico PCA
plt.figure(figsize=(10, 6))
plt.scatter(
    pca_result[:, 0], 
    pca_result[:, 1], 
    c=df_brazil['experience_level_code'], 
    cmap='viridis', 
    edgecolor='k', 
    s=100
)
plt.colorbar(label='Nível de Conhecimento (Codificado)')
plt.title('PCA dos Dados de Salário no Brasil', fontsize=14)
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
df = pd.read_csv("data_science_salaries.csv")

# Filtrar apenas os dados do Brasil
df_brazil = df[df['employee_residence'] == 'Brazil']

# Criar o gráfico de dispersão categórico
plt.figure(figsize=(12, 6))
for level in df_brazil['experience_level'].unique():
    subset = df_brazil[df_brazil['experience_level'] == level]
    plt.scatter(
        subset['job_title'], 
        subset['salary_in_usd'], 
        label=level, 
        alpha=0.7
    )

# Adicionar legendas e títulos
plt.title('Salários no Brasil por Nível de Conhecimento', fontsize=14)
plt.xlabel('Cargo', fontsize=12)
plt.ylabel('Salário em USD', fontsize=12)
plt.legend(title='Nível de Conhecimento')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
df = pd.read_csv("data_science_salaries.csv")

# Criar o gráfico de dispersão categórico
plt.figure(figsize=(16, 8))
for level in df['experience_level'].unique():
    subset = df[df['experience_level'] == level]
    plt.scatter(
        subset['employee_residence'], 
        subset['salary_in_usd'], 
        label=level, 
        alpha=0.7
    )

# Adicionar legendas e títulos
plt.title('Salários por Residência do Empregado e Nível de Experiência', fontsize=14)
plt.xlabel('Residência do Empregado', fontsize=12)
plt.ylabel('Salário em USD (Escala Original)', fontsize=12)
plt.legend(title='Nível de Experiência')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
