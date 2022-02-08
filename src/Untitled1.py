#!/usr/bin/env python
# coding: utf-8

# In[25]:


from matplotlib import pyplot as plt
from scipy.signal import lfilter, savgol_filter
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os


# In[32]:



path = os.path.dirname(os.path.abspath('')) + '/src/logs/carracing/'
list_dir = ['rnn/jsons/test_loss.json', 'rnn/jsons/train_loss.json']


# In[33]:


n = 10  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1


# In[28]:


color = {6: 'firebrick', 7: 'tomato', 2: 'peru', 3: 'gold', 0: 'dodgerblue', 5: 'springgreen', 6: 'indigo', 1: 'deeppink'}


# In[81]:


plt.figure()
labels = ['Teste', 'Treino']
for c, directory in tqdm(enumerate(list_dir), total=len(list_dir)):
    with open(path+directory) as f:
        data = json.load(f)


    
    y = pd.DataFrame(data).iloc[:, 2].to_numpy()
    
    episodes = pd.DataFrame(data).iloc[:, 1].to_numpy()
    smooth = savgol_filter(y,window_length = 5, polyorder = 1)
    
    plt.plot(episodes, smooth, color=color[c], linestyle='-', linewidth=2, label=labels[c])
    plt.plot(episodes, y, color=color[c], linestyle='-', linewidth=1, alpha=0.25)  # label='Real Rewards' 
    
    


# In[63]:


plt.title('Função de Perda por Época da RNN e MDN', size=16)
plt.legend(loc=4, prop={'size': 12})
plt.xlabel('Época')
plt.ylabel('Perda')
plt.xlim([0, 55])
plt.ylim([1.1, 1.6])
plt.grid()
plt.show()


# In[ ]:




