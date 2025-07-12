# Segmentação de Clientes de um Supermercado

Um supermercado, através de cartões de fidelidade, possui alguns dados básicos sobre seus clientes, como idade, gênero, renda anual e pontuação de gastos. Tal pontuação é algo que o supermercado atribui ao cliente com base em parâmetros definidos, como comportamento do cliente e dados de compra. 
O projeto atual visa segmentar pos clientes do supermercado através da técnica de clusterização para que o supermercado possa definir estratégias específicas para cada cliente visando aumentar sua receita. 

A base de dados original foi obtida através do Kaggle - [Link original para o dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

Abaixo será detalhado as etapas e considerações a respeito desse projeto de ciência dos dados.

## **Análise Exploratória**
- A análise exploratória do projeto foi feita de 2 maneiras. Primeiramente através do notebook [EDA_Analise_Exploratoria](notebooks/01_Analise_Exploratoria.ipynb) para que fosse feito o conhecimento inicial da base com comandos do pandas, matplotlib e seaborn, e posteriormente com o apoio da biblioteca ydata_profiling gerando o [ProfileReport](relatorios/EDA_supermercado.html) que foi extraído e salvo dentro da pasta "relatorios e imagens" podendo ser acessado mesmo sem necessitar rodar os notebooks .ipynb.

## **Modelo de Clusterização**
- O projeto de projeto de segmentação de clientes ocorreu através da biblioteca KMeans do Scikit-Learn. 2 notebooks foram desenvolvidos no decorrer desse projeto:
  - [Projeto_Clusterizacao_sem_PCA](notebooks/02B.projeto_clusterizacao_sem_PCA_final.ipynb)
  - [Projeto_Clusterizacao_com_PCA](notebooks/02B.projeto_clusterizacao_com_PCA_final.ipynb)
- Para preparar os dados para clusterização, foi utilizado um pipeline de preprocessamento para transformar a coluna categória 'Gender' através do one_hot_encoder e para normalizar os dados das colunas numéricas através do StandardScaler. 
- Para que fossem definidos o número de clusters mais adequado para o modelo KMeans, foi gerado gráficos através de 2 métodos (Elbow Method e Silhouette Score) onde foi iterado o número de clusters de 2 a 10:
  - *Elbow Method*: soma da distância quadrada de cada ponto para o centroide de seu cluster (inercia). O número de clusters mais indicado é onde vemos o "cotovelo" no gráfico, ou quando o resultado da 'inercia' começa a ter uma queda menos acentuada no gráfico, ou seja, o aumento do número de clusters começa a gerar ganhos marginais apenas.
  - *Silhouette Score*: dentro de cada cluster ele calcula a distância média de cada ponto, e compara com a distânia do ponto pro cluster mais próximo que ele não pertence. É tomado a média dessas distâncias, e o número de clusters mais indicado é quando percebemos o "silhouette score" mais alto no gráfico antes da primeira queda.
  - *Decisão do número de clusters*: é decidido o número de cluster interpretando os 2 gráficos e enxergando um número que possa ser comum e indicado em ambas análises. Em ambos os notebooks trabalhamos decidimos trabalhar com 5 clusters, mas poderiam ser 4 clusters também.

<div align="center"> <img src="relatorios/Definicao dos Clusters - Elbow Method e Silhouette Score.png" title="Pairplot" height="700"/> </div>

- Também foi utilizado o método de *Principal Component Analysis (PCA)* no notebook [Projeto_Clusterizacao_com_PCA](notebooks/02B.projeto_clusterizacao_com_PCA_final.ipynb) dentro do pipeline do modelo de clusterização para entender se os resultados com a redução de dimensionalidade seriam muito diferentes daquele sem o método de PCA. Concluímos que incluindo PCA no projeto ainda é recomendado manter os 5 clusters e a segmentação dos clientes ficou muito similar ao notebook sem o método PCA incluído.
  - OBS: Para determinar quantos componentes resultariam da PCA foi utilizado o gráfico *Scree Plot*, que demonstra qual é a variância explicada a medida que aumentamos o número de componentes da PCA em comparação com variância original dos dados.

<div align="center"> <img src="relatorios/Scree Plot - Variancia Acumulada Explicada.png" title="Scree Plot" height="400"/> </div>

- Tendo determinado o número de clusters mais adequado e o de componentes da PCA, o modelo de clusterização é criado e treinado através de um pipeline entre todas as etapas de preprocessamento dos dados, do PCA e do modelo KMeans. É definido à qual cluster cada cliente do dataframe pertence, algumas análises gráficas foram tomadas para podermos visualizar os diferentes grupos de clientes.
- Nos notebooks do projeto pudemos notar também que o gênero que cada cliente pertence não tem importância na determinação dos clusters. Por conta disso os gráficos abaixo não demonstração dados referente à o Genero.
- Abaixo teremos gráficos de Pairplot e Boxplot onde poderemos ver como cada cluster se diferencia entre si nas 3 variáveis mais significativas para o projeto, que são a renda anual do cliente, sua idade, e seu score de gastos. 

<div align="center"> <img src="relatorios/Pairplot Clusters.png" title="Pairplot" height="600"/> </div>

<div align="center"> <img src="relatorios/Boxplot dos Clusters - Idade - Renda - Score.png" title="Boxplot" height="600"/> </div>


## Definição dos Clusters

| Pontuação de Gastos | Renda | Idade | Cluster |
|---------------------|-------|-------|---------|
| Alta                | Alta  | Jovem/Adulto | 0       |
| Moderada            | Moderada | Adulto/Idoso | 1       |
| Alta                | Baixa | Jovem | 2       |
| Baixa               | Alta  | Adulto | 3       |
| Moderada            | Moderada | Jovem/Adulto | 4       |

- Por fim o dataframe original com a coluna à qual cada cliente pertence foi extraído e pode ser verificado na pasta "dados", com nome [Mall_Customers_clustered](dados/Mall_Customers_clustered.csv)
- Caso queiram consultar o modelo criado nesse projeto, os mesmos se encontram na pasta "modelos", podendo ser encontrado o [Modelo de Clusterização sem PCA](modelos/pipeline_preprocessamento_clustering.pkl) e [Modelo de Clusterização com PCA](modelos/pipeline_preprocessamento_pca_clustering.pkl)


## Organização de pastas e arquivos projeto

```
├── .env               <- Arquivo de variáveis de ambiente (não versionar)
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git
├── ambiente.yml       <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE            <- Licença de código aberto se uma for escolhida
├── README.md          <- README principal para desenvolvedores que usam este projeto.
|
├── dados              <- Arquivos de dados para o projeto.
|
├── modelos            <- Modelos treinados e serializados, previsões de modelos ou resumos de modelos
|
├── notebooks          <- Cadernos Jupyter onde foi desenvolvido o projeto.
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py  <- Torna um módulo Python
|      ├── config.py    <- Configurações básicas do projeto
|      └── graficos.py  <- Scripts para criar visualizações exploratórias e orientadas a resultados
|
├── relatorios         <- Análises geradas em HTML, PDF, gráficos e figuras gerados para serem usados em relatórios
```
