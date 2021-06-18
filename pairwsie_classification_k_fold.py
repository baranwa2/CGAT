import glob
import timeit
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold


### List of all pairwise classes

f1 = ['CP', 'CP', 'CP', 'CP', 'CP', 'IPMN', 'IPMN', 'IPMN', 'IPMN', 'MCN', 'MCN', 'MCN', 'PanIN', 'PanIN', 'PDAC']
f2 = ['IPMN', 'MCN', 'PanIN', 'PDAC', 'SpecDxIPMN', 'MCN', 'PanIN', 'PDAC', 'SpecDxIPMN', 'PanIN', 'PDAC', 'SpecDxIPMN', 'PDAC', 'SpecDxIPMN', 'SpecDxIPMN']

### Location of data files/ need to be replaed with the correct source
dir_input = ('/gdrive/My Drive/Tumor_Grading/')

class GroupPredictor(nn.Module):
  def __init__(self):
    super(GroupPredictor, self).__init__()
    self.embed_atom  = nn.Embedding(n_classes, dim)
    self.W_atom      = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
    self.W_attention = nn.Linear(dim, dim)
    self.w           = nn.Parameter(torch.zeros(dim))
    self.W_property  = nn.Linear(dim, 2)
      
  def update(self, xs, adjacency, i):
    hs = torch.relu(self.W_atom[i](xs))
    return torch.matmul(adjacency, hs)


  def self_attention(self, h):
    x = self.W_attention(h)
    m = x.size()[0]
    
    d     = torch.tanh(x)
    alpha = torch.matmul(d,self.w).view(m,1)
    
    p = torch.softmax(alpha,0)
    s = torch.matmul(torch.t(x),p).view(1,-1)
    
    return s, p
        
    
  def forward(self, adjacency, feature):
    
    x = self.embed_atom(feature)
    for i in range(layer):
      x = self.update(x, adjacency, i)
        
    s, p  = self.self_attention(x)
    return self.W_property(s), p
    
  def __call__(self, data_batch, train=True):
      
    feature = torch.LongTensor(data_batch[-1]).view(-1)
    
    adjacency, t_interaction = torch.FloatTensor(data_batch[:-2][0]), data_batch[-2]
    
    z_interaction, attn_wts = self.forward(adjacency, feature)
    
    if train:
      loss = F.cross_entropy(z_interaction, t_interaction)
      return loss
    else:
      zs = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
      ts = int(t_interaction.to('cpu').data[0].numpy())
      return zs, ts, attn_wts


class Trainer(object):
  def __init__(self, model):
    self.model = model
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

  def train(self, dataset):
    np.random.shuffle(dataset)
    loss_total = 0
    for data in dataset:
      loss = self.model(data)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      loss_total += loss.to('cpu').data.numpy()
    return loss_total


class Tester(object):
  def __init__(self, model):
    self.model = model

  def test(self, dataset):
    z_list, t_list, a_list = [], [], []
    for data in dataset:
      z, t, attn_wts = self.model(data, train=False)
      z_list.append(z)
      t_list.append(t)
      a_list.append(attn_wts)

    score_list, label_list = [], []
    for z in z_list:
      score_list.append(z[1])
      label_list.append(np.argmax(z))

    auc       = roc_auc_score(t_list, score_list)
    precision = precision_score(t_list, label_list, zero_division=1)
    recall    = recall_score(t_list, label_list, zero_division=1)

    return auc, precision, recall, attn_wts

dim            = 30
layer          = 2
lr             = 1e-3
lr_decay       = 0.75
decay_interval = 20
iteration      = 100
n_classes      = 3

(dim, layer, decay_interval, iteration) = map(int, [dim, layer, decay_interval, iteration])
lr, lr_decay                            = map(float, [lr, lr_decay])

cell_dict             = {}
cell_dict['"Tumor"']  = 0
cell_dict['"TIL"']    = 1
cell_dict['"Treg"']   = 2

max_x, max_y = 1392, 1040

num_k_folds  = 5
kfold        = KFold(num_k_folds, True, 1)
kNN          = 20
knn          = NearestNeighbors(n_neighbors=kNN)

# Directiories where trained models and results need to be saved
dir_results = ('/gdrive/My Drive/Tumor_Grading/results/')
dir_models  = ('/gdrive/My Drive/Tumor_Grading/models/')

for file_ind in range(15):

  # Creating hold-out test dataset
  adjacencies, features, properties = [], [], []

  t_name   = 'Edited_' + f1[file_ind]
  f_names1 = glob.glob(dir_input+'TestingData/'+t_name+"/*")
  t_name   = 'Edited_' + f2[file_ind]
  f_names2 = glob.glob(dir_input+'TestingData/'+t_name+"/*")

  for f in f_names1:
    with open(f, 'r') as freader:
      data_list = freader.read().strip().split('\n')
      x_pix, y_pix, cID = [], [], []
      for k in range(len(data_list)-1):
        x, y, _, ID = data_list[k+1].strip().split(',')
        if '""' not in ID:
          x_pix.append(int(x)/max_x)
          y_pix.append(int(y)/max_y)
          cID.append(cell_dict[ID])

      X = np.concatenate((np.reshape(np.array(x_pix),(-1,1)), np.reshape(np.array(y_pix),(-1,1))), axis=1)
      knn.fit(X)

      A = np.zeros((len(X),len(X)))
      for i in range(len(X)):
        _, inds = knn.kneighbors(X[i].reshape((1,-1)))
        A[i,inds[0]] = 1

      adjacency = 0.5*(A + np.transpose(A))
      adjacency[adjacency>0] = 1

      degree     = sum(adjacency)
      d_half     = np.sqrt(np.diag(degree))
      d_half_inv = np.linalg.inv(d_half)
      adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))

      adjacencies.append(adjacency)
      features.append(np.array(cID).reshape((-1,1)))
      properties.append(np.array([int(1)]))

  for f in f_names2:
    with open(f, 'r') as freader:
      data_list = freader.read().strip().split('\n')
      x_pix, y_pix, cID = [], [], []
      for k in range(len(data_list)-1):
        x, y, _, ID = data_list[k+1].strip().split(',')
        if '""' not in ID:
          x_pix.append(int(x)/max_x)
          y_pix.append(int(y)/max_y)
          cID.append(cell_dict[ID])

      X = np.concatenate((np.reshape(np.array(x_pix),(-1,1)), np.reshape(np.array(y_pix),(-1,1))), axis=1)
      knn.fit(X)

      A = np.zeros((len(X),len(X)))
      for i in range(len(X)):
        _, inds = knn.kneighbors(X[i].reshape((1,-1)))
        A[i,inds[0]] = 1

      adjacency = 0.5*(A + np.transpose(A))
      adjacency[adjacency>0] = 1

      degree     = sum(adjacency)
      d_half     = np.sqrt(np.diag(degree))
      d_half_inv = np.linalg.inv(d_half)
      adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))

      adjacencies.append(adjacency)
      features.append(np.array(cID).reshape((-1,1)))
      properties.append(np.array([int(0)]))

  t_properties = [torch.LongTensor(d) for d in properties]
  dataset_test = list(zip(adjacencies, t_properties, features))
  random.shuffle(dataset_test)





  #Creating train and cv datasets
  adjacencies, features, properties = [], [], []
  f_names1, f_names2                = [], []

  t_name   = 'Edited_' + f1[file_ind]
  f_names1 = glob.glob(dir_input+'TrainingData/'+t_name+"/*")
  t_name   = 'Edited_' + f2[file_ind]
  f_names2 = glob.glob(dir_input+'TrainingData/'+t_name+"/*")

  for f in f_names1:
    with open(f, 'r') as freader:
      data_list = freader.read().strip().split('\n')
      x_pix, y_pix, cID = [], [], []
      for k in range(len(data_list)-1):
        x, y, _, ID = data_list[k+1].strip().split(',')
        if '""' not in ID:
          x_pix.append(int(x)/max_x)
          y_pix.append(int(y)/max_y)
          cID.append(cell_dict[ID])

      X = np.concatenate((np.reshape(np.array(x_pix),(-1,1)), np.reshape(np.array(y_pix),(-1,1))), axis=1)
      knn.fit(X)

      A = np.zeros((len(X),len(X)))
      for i in range(len(X)):
        _, inds = knn.kneighbors(X[i].reshape((1,-1)))
        A[i,inds[0]] = 1

      adjacency = 0.5*(A + np.transpose(A))
      adjacency[adjacency>0] = 1

      degree     = sum(adjacency)
      d_half     = np.sqrt(np.diag(degree))
      d_half_inv = np.linalg.inv(d_half)
      adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))

      adjacencies.append(adjacency)
      features.append(np.array(cID).reshape((-1,1)))
      properties.append(np.array([int(1)]))

  for f in f_names2:
    with open(f, 'r') as freader:
      data_list = freader.read().strip().split('\n')
      x_pix, y_pix, cID = [], [], []
      for k in range(len(data_list)-1):
        x, y, _, ID = data_list[k+1].strip().split(',')
        if '""' not in ID:
          x_pix.append(int(x)/max_x)
          y_pix.append(int(y)/max_y)
          cID.append(cell_dict[ID])

      X = np.concatenate((np.reshape(np.array(x_pix),(-1,1)), np.reshape(np.array(y_pix),(-1,1))), axis=1)
      knn.fit(X)

      A = np.zeros((len(X),len(X)))
      for i in range(len(X)):
        _, inds = knn.kneighbors(X[i].reshape((1,-1)))
        A[i,inds[0]] = 1

      adjacency = 0.5*(A + np.transpose(A))
      adjacency[adjacency>0] = 1

      degree     = sum(adjacency)
      d_half     = np.sqrt(np.diag(degree))
      d_half_inv = np.linalg.inv(d_half)
      adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))

      adjacencies.append(adjacency)
      features.append(np.array(cID).reshape((-1,1)))
      properties.append(np.array([int(0)]))

  t_properties = [torch.LongTensor(d) for d in properties]
  dataset = list(zip(adjacencies, t_properties, features))
  random.shuffle(dataset)
  random.shuffle(dataset)
  random.shuffle(dataset)

  count_fold = 1
  for train, test in kfold.split(dataset):
    dataset_train = [dataset[i] for i in train]
    dataset_cv    = [dataset[i] for i in test]

    print('Fold identity : '+str(count_fold))
    count_fold += 1

    torch.manual_seed(1234)

    model   = GroupPredictor()
    trainer = Trainer(model)
    tester  = Tester(model)

    file_result = dir_results + f1[file_ind] + '_' + f2[file_ind] + '/fold_' + str(count_fold-1) + '.txt'
    with open(file_result, 'w') as file_reader:
      file_reader.write('Epoch \t Time(sec) \t Loss_train \t AUC_CV \t Precision_CV \t Recall_CV \t AUC_test \t Precision_test \t Recall_test\n')

    start = timeit.default_timer()

    print('Differentiating b/w ' + f1[file_ind] + ' and ' + f2[file_ind] + ' for fold ' + str(count_fold-1))
    print('Training...')
    print('Epoch \t Time(sec) \t Loss_train \t AUC_CV \t Precision_CV \t Recall_CV \t AUC_test \t Precision_test \t Recall_test')

    for epoch in range(iteration):
      if (epoch+1) % decay_interval == 0:
        trainer.optimizer.param_groups[0]['lr'] *= lr_decay

      loss    = trainer.train(dataset_train)

      auc_cv, precision_cv, recall_cv, _       = tester.test(dataset_cv)
      auc_test, precision_test, recall_test, _ = tester.test(dataset_test)

      lr_rate = trainer.optimizer.param_groups[0]['lr']

      end  = timeit.default_timer()
      time = end - start

      print('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' %(epoch+1, time, loss, auc_cv, precision_cv, recall_cv, auc_test, precision_test, recall_test))

      AUCs = [epoch+1, time, loss, auc_cv, precision_cv, recall_cv, auc_test, precision_test, recall_test]
      with open(file_result, 'a') as file_reader:
        file_reader.write('\t'.join(map(str, AUCs)) + '\n')

      file_model = dir_models + f1[file_ind] + '_' + f2[file_ind] + '/fold_' + str(count_fold-1) + '_epoch_' + str(epoch+1) 
      torch.save(model.state_dict(), file_model)