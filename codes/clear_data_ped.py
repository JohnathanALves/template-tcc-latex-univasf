#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('data/pedagogiaDataframe.csv')


# In[3]:


df.head()


# Vamos manter apenas os dados utilizados pelos algoritmos de aprendizagem

# In[4]:


newIdx = ['VAR01',
          'VAR04',
          'VAR05',
          'VAR08',
          'VAR13a',
          'VAR13b',
          'VAR13c',
          'VAR16',
          'VAR18',
          'VAR19',
          'VAR20',
          'VAR21',
          'VAR22',
          'VAR23',
          'VAR27',
          'VAR28',
          'VAR31',
          'EVADIU'
          ]

df = df[newIdx]


# Adapta os nomes das variáveis

# In[5]:


df.columns = ['VAR01',
              'VAR02',
              'VAR03',
              'VAR04',
              'VAR05a',
              'VAR05b',
              'VAR05c',
              'VAR06',
              'VAR07',
              'VAR08',
              'VAR09',
              'VAR10',
              'VAR11',
              'VAR12',
              'VAR12',
              'VAR14',
              'VAR15',
              'EVADIU'
              ]


# In[6]:


df.describe().transpose()[['min', 'mean', '50%', 'max']]


# In[7]:


df.info()


# Mostrar a distribuição dos dados em relação à variável alvo

# In[8]:


sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.0)
ax = sns.countplot(x='EVADIU',
    data=pd.DataFrame(
    df['EVADIU'].replace({0: 'Não evadiu', 1: 'Evadiu'})),
    palette="cubehelix")
plt.ylabel('Quantidade')
plt.xlabel('')

total = len(df['EVADIU'])
plt.ylim(0, (total / 2 + 300))

for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = (p.get_x() + p.get_width() / 2) - 0.15
    y = p.get_height() + 20.0
    ax.annotate(percentage, (x, y))

plt.savefig(fname='images/barplot_pedagogia.svg', format='svg')


# # Classificação

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X = df.drop(['EVADIU'],axis=1)
y = df['EVADIU']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)


# #### Relatório de balanceamento de classes

# Função auxiliar para plotagem

# In[11]:


def plot_class_balance(data, xlabel, ylabel, name):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=2.0)
    ax = sns.countplot(data.replace({0: 'Não evadiu', 1: 'Evadiu'}), palette="cubehelix")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    total = len(data)
    plt.ylim(0, (total / 2 + 250))
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = (p.get_x() + p.get_width() / 2) - 0.15
        y = p.get_height() + 20.0
        ax.annotate(percentage, (x, y))
    plt.savefig(fname=name, format='svg')


# Dados de treinamento

# In[12]:


plot_class_balance(y_train, '', 'Quantidade', 'images/barplot_pedagogia_treinamento.svg')


# Dados de teste

# In[13]:


plot_class_balance(y_test, '', 'Quantidade', 'images/barplot_pedagogia_testes.svg')


# ### Funções auxiliares para gerar métricas e relatórios

# In[14]:


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# In[15]:


def plot_confusion_matrix(y_true, y_pred, classes, name,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.set_style('white')
    sns.set_context("paper", font_scale=1.5)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='',
           ylabel='Valores reais',
           xlabel='Valores preditos')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    plt.savefig(fname=name, format='svg')
    return ax


np.set_printoptions(precision=2)


# In[16]:


def custom_classification_report(y_true, y_pred, title='Relatórios das métricas de classificação'):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(title + '\n')

    print('Acurácia: \t\t%5.4f' % ((tp + tn) / (tp + tn + fp + fn)))

    print('Precisão: \t\t%5.4f' % (tp / (tp + fp)))

    print('Sensibilidade: \t\t%5.4f' % (tp / (tp + fn)))

    print('Especificidade: \t%5.4f' % (tn / (tn + fp)))

    print('TFP: \t\t\t%5.4f' % (fp / (fp + tn)))

    print('TFN: \t\t\t%5.4f' % (fn / (fn + tp)))


# In[17]:


def make_accs_knn(x_train, x_test, y_train, y_test, test_values):
    accs = []
    best_k = 1
    maxi = 0.
    for k in test_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(x_train, y_train)
        acc = knn.score(x_test, y_test)
        if(acc > maxi):
            maxi = acc
            best_k = k
        accs.append(acc)
    return accs, maxi, best_k


# In[18]:


def plot_acc(x_train, x_test, y_train, y_test, test_values, name):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    accs, maxi, best_k = make_accs_knn(x_train, x_test, y_train, y_test, test_values)
    plt.plot(test_values, accs, 'k')
    plt.xlabel('K')
    plt.ylabel('Acurácia')
    plt.savefig(fname=name, format='svg')
    plt.show()
    print('Maior acurácia: '+ str(maxi))
    print('Melhor k: '+ str(best_k))


# ## KNN

# In[19]:


from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis


# Testando os valores de K entre 0 e 203

# In[20]:


plot_acc(X_train, X_test, y_train, y_test, list(range(1, 203, 2)), 'images/knn_neigh_ped.svg')


# Na tentativa de melhorar a previsão com KNN, iremos normalizar os dados.

# In[21]:


from sklearn import preprocessing


# In[22]:


# Get column names first
names = X_train.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=names)

X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=names)


# Realizando mesmo teste com k entre 0 e 203

# In[23]:


plot_acc(X_train_scaled, X_test_scaled, y_train, y_test, list(range(1, 203, 2)), 'images/knn_neigh_norm_ped.svg')


# O melhor resultado foi para K = 3 sem normalização, vamos verificar todas as variáveis.

# In[24]:


knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)


# In[25]:


knn.fit(X_train, y_train)


# In[26]:


predictions = knn.predict(X_test)


# In[27]:


custom_classification_report(y_test, predictions, 'Relatório de métricas para o KNN com dados não normalizados')


# In[28]:


plot_confusion_matrix(y_test, predictions, name='images/cm_knn_ped.svg',
                      classes=np.array(['Evadiu', 'Não evadiu']))
plt.savefig(fname='images/test.svg', format='svg')


# ### Decision Tree

# In[29]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


# In[30]:


treemodel = DecisionTreeClassifier(criterion='entropy')
treemodel.fit(X_train,y_train)


# In[31]:


predictions = treemodel.predict(X_test)


# In[32]:


custom_classification_report(y_test, predictions, 'Relatório de Métricas Para Árvore de Decisão')


# In[33]:


plot_confusion_matrix(y_test, predictions, name='images/cm_tree_ped.svg',
                      classes=np.array(['Evadiu', 'Não evadiu']))


# In[34]:


export_graphviz(treemodel, out_file='images/ped_tree.dot',
                           max_depth=3,
                           feature_names=X.columns,
                           class_names=['Não evadiu', 'Evadiu'],
                           filled=True, rounded=True,
                           special_characters=True)

dot_data = export_graphviz(treemodel, out_file=None,
                           max_depth=3,
                           feature_names=X.columns,
                           class_names=['Não evadiu', 'Evadiu'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)

graph


# #### Importância de variáveis utilizando SelectFromModel

# In[35]:


treemodel.feature_importances_


# In[36]:


from sklearn.feature_selection import SelectFromModel


# In[37]:


treemodel = DecisionTreeClassifier(criterion='entropy')
treemodel.fit(X, y)
model = SelectFromModel(treemodel, prefit=True)


# In[38]:


X_new = model.transform(X)
X_new.shape


# VAR07 é a mais importante para a árvore de decisão

# ### Regressão Logistica

# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train)


# In[41]:


predictions = logmodel.predict(X_test)


# In[42]:


custom_classification_report(y_test, predictions, 'Relatório de Métricas Para Regressão Logistica')


# In[43]:


plot_confusion_matrix(y_test, predictions, name='images/cm_rl_ped.svg',
                      classes=np.array(['Evadiu', 'Não evadiu']))


# # Testes sem a variável `VAR07`, ela será removida por ser uma variável enviesada.

# In[44]:


df = df.drop('VAR07',axis=1)
df.head()


# In[45]:


X = df.drop(['EVADIU'],axis=1)
y = df['EVADIU']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# ### Testes KNN

# Escolha do parametro K novamente

# In[46]:


plot_acc(X_train, X_test, y_train, y_test, list(range(1, 203, 2)), 'images/knn_neigh_ped_no_var07.svg')


# Normaliza os dados sem variável `VAR07`

# In[47]:


# Get column names first
names = X_train.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=names)

X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=names)


# In[48]:


plot_acc(X_train_scaled, X_test_scaled, y_train, y_test, list(
    range(1, 203, 2)), 'images/knn_neigh_norm_ped_no_var07.svg')


# Resultado: K = 1 e dados não normalizados

# In[49]:


knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(X_train, y_train)


# Executando e gerando os relatórios para o KNN

# In[50]:


predictions = knn.predict(X_test)


# In[51]:


custom_classification_report(y_test, predictions, 'Relatório de métricas para o KNN com dados não normalizados')


# In[52]:


plot_confusion_matrix(y_test, predictions, name='images/cm_knn_ped_no_var07.svg',
                      classes=np.array(['Evadiu', 'Não evadiu']))


# ### Testes com Árvore de Decisão

# In[53]:


treemodel = DecisionTreeClassifier(criterion='entropy')
treemodel.fit(X_train,y_train)


# In[54]:


predictions = treemodel.predict(X_test)


# In[55]:


custom_classification_report(y_test, predictions, 'Relatório de Métricas Para Árvore de Decisão')


# In[56]:


plot_confusion_matrix(y_test, predictions, name='images/cm_tree_ped_no_var07.svg',
                      classes=np.array(['Evadiu', 'Não evadiu']))


# In[57]:


export_graphviz(treemodel, out_file='images/ped_tree_no_var07.dot',
                           max_depth=3,
                           feature_names=X.columns,
                           class_names=['Não evadiu', 'Evadiu'],
                           filled=True, rounded=True,
                           special_characters=True)

dot_data = export_graphviz(treemodel, out_file=None,
                           max_depth=3,
                           feature_names=X.columns,
                           class_names=['Não evadiu', 'Evadiu'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph


# In[58]:


treemodel.feature_importances_


# In[59]:


treemodel = DecisionTreeClassifier(criterion='entropy')
treemodel.fit(X, y)
model = SelectFromModel(treemodel, prefit=True)


# In[60]:


X_new = model.transform(X)
X_new.shape


# ### Testes com RL

# In[61]:


logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train)


# In[62]:


predictions = logmodel.predict(X_test)


# In[63]:


custom_classification_report(y_test, predictions, 'Relatório de Métricas Para Regressão Logistica')


# In[64]:


plot_confusion_matrix(y_test, predictions, name='images/cm_rl_ped_no_var07.svg',
                      classes=np.array(['Evadiu', 'Não evadiu']))

