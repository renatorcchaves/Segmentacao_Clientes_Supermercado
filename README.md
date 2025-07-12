# Segmentação de Clientes de um Supermercado

Um supermercado, através de cartões de fidelidade, possui alguns dados básicos sobre seus clientes, como idade, gênero, renda anual e pontuação de gastos. Tal pontuação é algo que o supermercado atribui ao cliente com base em parâmetros definidos, como comportamento do cliente e dados de compra. 
O projeto atual visa segmentar pos clientes do supermercado através da técnica de clusterização para que o supermercado possa definir estratégias específicas para cada cliente visando aumentar sua receita. 

A base de dados original foi obtida através do Kaggle - [Link original para o dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

Abaixo será detalhado as etapas e considerações a respeito desse projeto de ciência dos dados.

## **Análise Exploratória**
- A análise exploratória do projeto foi feita de 2 maneiras. Primeiramente através do notebook [EDA_Analise_Exploratoria](notebooks/01_Analise_Exploratoria.ipynb) para que fosse feito o conhecimento inicial da base com comandos do pandas, matplotlib e seaborn, e posteriormente com o apoio da biblioteca ydata_profiling gerando o [ProfileReport](relatorios%20e%20imagens/EDA_supermercado.html) que foi extraído e salvo dentro da pasta "relatorios e imagens" podendo ser acessado mesmo sem necessitar rodar os notebooks .ipynb.

## **Modelo de Clusterização**
- O projeto de projeto de segmentação de clientes ocorreu através da biblioteca KMeans do Scikit-Learn.
- Para preparar os dados para essa biblioteca, foi utilizado um pipeline de preprocessamento para transformar a coluna categória 'Gender' através do one_hot_encoder e para normalizar os dados das colunas numéricas através do StandardScaler. 
- Para que fossm definidos o número de clusters mais adequado para o modelo KMeans, foi gerado gráficos através de 2 métodos (Elbow Method e Silhouette SScore) onde foi iterado o número de clusters de 2 a 10:
  - *Elbow Method*: soma da distância quadrada de cada ponto para o centroide de seu cluster (inercia). O número de clusters mais indicado é onde vemos o "cotovelo" no gráfico, ou quando o resultado da 'inercia' começa a ter uma queda menos acentuada no gráfico, ou seja, o aumento do número de clusters começa a gerar ganhos marginais apenas.
  - *Silhouette Score*: dentro de cada cluster ele calcula a distância média de cada ponto, e compara com a distânia do ponto pro cluster mais próximo que ele não pertence. É tomado a média dessas distâncias, e o número de clusters mais indicado é quando percebemos o "silhouette score" mais alto no gráfico antes da primeira queda.
  - *Decisão do número de clusters*: é decidido o número de cluster interpretando os 2 gráficos e enxergando um número que possa ser comum e indicado em ambas análises.
**[Elbow%20Method%20e%20Silhoutte%20Score](relatorios%20e%20imagens/Definicao%20dos%20Clusters%20-%20Elbow%20Method%20e%20Silhouette%20Score.png)**

- Também foi utilizado o método de *Principal Component Analysis (PCA)* no notebook [Projeto_Clusterizacao_com_PCA](notebooks/02B.projeto_clusterizacao_com_PCA_final.ipynb) dentro do pipeline do modelo de clusterização para entender se os resultados com a redução de dimensionalidade seriam muito diferentes daquele sem o método de PCA.
  - Para determinar quantos componentes resultariam da PCA foi utilizado o gráfico *Scree Plot*, que demonstra qual é a variância explicada a medida que aumentamos o número de componentes resultantes da PCA em comparação com variância original dos dados.
**[Scree%20Plot](Scree%20Plot%20-%20Variancia%20Acumulada%20Explicada.png)**

- Tendo determinado o número de clusters mais adequado e o de componentes da PCA, o modelo de clusterização é criado e treinado através de um pipeline entre todas as etapas de preprocessamento dos dados, do PCA e do modelo KMeans. É definido à qual cluster cada cliente do dataframe pertence, algumas análises gráficas foram tomadas para podermos visualizar os diferentes grupos de clientes.
- Podemos notar também que o gênero que cada cliente pertence não tem importância na determinação dos clusters, que foram mais influenciados para renda do nliente, sua idade, e seu score de gastos.

**[Pairplot](relatorios e imagens/Pairplot Clusters.png)**
- **[Boxplot%20dos%20Clusters](relatorios%20e%20imagens/Boxplot%20dos%20Clusters%20-%20Idade%20-%20Renda%20-%20Score.png)**

Cada cluster pode ser dividido da seguinte maneira:
| Pontuação de Gastos | Renda | Idade | Cluster |
|---------------------|-------|-------|---------|
| Moderada                | Moderada  | Adultos/Idosos | 0       |
| Moderada            | Moderada | Jovens/Adultos | 1       |
| Baixa                | Alta | Adultos | 2       |
| Alta               | Baixa  | Jovens | 3       |
| Alta            | Alta | Jovens/Adultos | 4       |

- A visualização de cada cliente plotando gráficos cruzando 2 dessas 3 variáveis entre si pode ser vista abaoxo:
[Visualização%20dos%20Clientes%20-%20Idade,%20Renda%20Anual%20e%20Score%20de%20Gastos](relatorios%20e%20imagens/Visualizacao%20dos%20Clusters%20-%20Idade%20-%20Renda%20-%20Score.png)

- Por fim o dataframe original com a coluna à qual cada cliente pertence foi extraído e pode ser verificado na pasta dados, com nome [Mall_Customers_clustered][dados/Mall_Customers_clustered.csv]
- Caso queiram consultar o modelo criado nesse projeto, os mesmos se encontram na pasta "modelos", podendo ser encontrado o [Modelo%20de%20Clusterização%20sem%20PCA](modelos/pipeline_preprocessamento_clustering.pkl) e [Modelo%20de%20Clusterização%20com%20PCA](modelos/pipeline_preprocessamento_pca_clustering.pkl)






## Organização do projeto

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
├── notebooks          <- Cadernos Jupyter. A convenção de nomenclatura é um número (para ordenação),
│                         as iniciais do criador e uma descrição curta separada por `-`, por exemplo
│                         `01-fb-exploracao-inicial-de-dados`.
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py  <- Torna um módulo Python
|      ├── config.py    <- Configurações básicas do projeto
|      └── graficos.py  <- Scripts para criar visualizações exploratórias e orientadas a resultados
|
├── referencias        <- Dicionários de dados, manuais e todos os outros materiais explicativos.
|
├── relatorios         <- Análises geradas em HTML, PDF, LaTeX, etc.
│   └── imagens        <- Gráficos e figuras gerados para serem usados em relatórios
```

## Configuração do ambiente

1. Faça o clone do repositório que será criado a partir deste modelo.

    ```bash
    git clone ENDERECO_DO_REPOSITORIO
    ```

2. Crie um ambiente virtual para o seu projeto utilizando o gerenciador de ambientes de sua preferência.

    a. Caso esteja utilizando o `conda`, exporte as dependências do ambiente para o arquivo `ambiente.yml`:

      ```bash
      conda env export > ambiente.yml
      ```

    b. Caso esteja utilizando outro gerenciador de ambientes, exporte as dependências
    para o arquivo `requirements.txt` ou outro formato de sua preferência. Adicione o
    arquivo ao controle de versão, removendo o arquivo `ambiente.yml`.

3. Verifique o arquivo `notebooks/01-fb-exemplo.ipynb` para exemplos
de uso do código.
4. Renomeie o arquivo `notebooks/01-fb-exemplo.ipynb` para um nome
mais apropriado ao seu projeto. E siga a convenção de nomenclatura para os demais
notebooks.
5. Remova arquivos de exemplo e adicione os arquivos de dados e notebooks do seu
projeto.
6. Verifique o arquivo `notebooks/src/config.py` para configurações básicas do projeto.
Modifique conforme necessário, adicionando ou removendo caminhos de arquivos e
diretórios.
7. Atualize o arquivo `referencias/01_dicionario_de_dados.md` com o dicionário de dados
do seu projeto.
8. Atualize o `README.md` com informações sobre o seu projeto.
9. Adicione uma licença ao projeto. Clique
[aqui](https://docs.github.com/pt/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
se precisar de ajuda para escolher uma licença.
10. Renomeie o arquivo `.env.exemplo` para `.env`
11. Adicione variáveis de ambiente sensíveis ao arquivo `.env`.

Por padrão, o arquivo `.gitignore` já está configurado para ignorar arquivos de dados e
arquivos de Notebook (para aqueles que usam ferramentas como
[Jupytext](https://jupytext.readthedocs.io/en/latest/) e similares). Adicione ou remova
outros arquivos e diretórios do `.gitignore` conforme necessário. Caso deseje adicionar
forçadamente um Notebook ao controle de versão, faça um commit forçado com o
comando `git add --force NOME_DO_ARQUIVO.ipynb`.

Para mais informações sobre como usar Git e GitHub, [clique aqui](https://cienciaprogramada.com.br/2021/09/guia-definitivo-git-github/). Sobre ambientes virtuais, [clique aqui](https://cienciaprogramada.com.br/2020/08/ambiente-virtual-projeto-python/).
