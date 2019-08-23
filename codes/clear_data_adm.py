#!/usr/bin/env python
# coding: utf-8

# # Preparação de dados - Administração

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_excel('data/BasesAdmPublicaFinal.xlsx', 'Plan2')
df.head()


# Corrige os valores na VAR27

# In[3]:


df['VAR27'] = df['VAR27'].apply(lambda x: x / 10000)


# In[4]:


df.info()


# Remover colunas desnecessárias e adiciona coluna alvo

# In[5]:


newIdx = ['Período',
          'ID da Disciplina',
          'ID do Aluno',
          'VAR01',
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
          ]

df = df[newIdx].dropna()
df.insert(len(df.columns), 'EVADIU', 0)
df.head()


# Todos os estudantes do curso

# In[6]:


studentsIds = df['ID do Aluno'].unique()
len(studentsIds)


# Quantidade de semestres

# In[7]:


semesters = df['Período'].unique()
len(semesters)


# Quantidade de disciplinas

# In[8]:


df['ID da Disciplina'].nunique()


# In[9]:


listOfStudents = []

for semester in semesters:
    courses = df[df['Período'] == semester]['ID da Disciplina'].unique()

    for course in courses:
        studentsInCourse = df[(df['Período'] == semester) &
                              (df['ID da Disciplina'] == course)]['ID do Aluno'].unique()

        diffStudents = np.setdiff1d(
            studentsIds, studentsInCourse, assume_unique=True)

        for student in diffStudents:
            listOfStudents.append(pd.Series(
                [semester,
                 course,
                 student,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 1 # Assumindo que os alunos que tiveram que ser inseridos foram os evadidos
                ],
                index=df.columns))


# In[10]:


expandedDf = df.append(listOfStudents, ignore_index=True)


# Agora vamos colocar a classe evadido em todas as aparições do aluno ao longo dos semestres

# In[11]:


evadedStudentsIds = expandedDf[expandedDf['EVADIU'] == 1]['ID do Aluno'].unique()
len(evadedStudentsIds)


# In[12]:


for student in evadedStudentsIds:
    expandedDf.loc[expandedDf['ID do Aluno'] == student, 'EVADIU'] = 1


# Salva versão expandida do _dataframe_

# In[13]:


expandedDf.to_csv('data/admDataframe.csv', index=False)

