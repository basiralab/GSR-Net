import torch
import numpy as np
import os
import scipy.io 

path= 'drive/My Drive/BRAIN_DATASET'
roi_str='ROI_FC.mat'

def pad_HR_adj(label, split):

  label=np.pad(label,((split,split),(split,split)),mode="constant")
  np.fill_diagonal(label,1)
  return torch.from_numpy(label).type(torch.FloatTensor)

def normalize_adj_torch(mx):
    # mx = mx.to_dense()
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

def unpad(data, split):
  
  idx_0 = data.shape[0]-split
  idx_1 = data.shape[1]-split
  # print(idx_0,idx_1)
  train = data[split:idx_0, split:idx_1]
  return train


def extract_data(subject,session_str, parcellation_str, subjects_roi):
  folder_path = os.path.join(path, str(subject), session_str, parcellation_str)
  roi_data = scipy.io.loadmat(os.path.join(folder_path, roi_str))
  roi=roi_data['r']

  #Replacing NaN values
  col_mean = np.nanmean(roi, axis=0)
  inds = np.where(np.isnan(roi))
  roi[inds] = 1

  #Taking the absolute values of the matrix
  roi = np.absolute(roi,dtype=np.float32)
#   roi = get_tensor(np.array(roi, dtype=np.float32))
  
  if parcellation_str == 'shen_268':
    roi = np.reshape(roi,(1,268,268))
  else:
    roi = np.reshape(roi,(1,160,160))
  
  if subject==25629:
    subjects_roi = roi
  else:
    subjects_roi = np.concatenate((subjects_roi,roi),axis=0)

  return subjects_roi

def load_data(start_value,end_value):
   
  subjects_label = np.zeros((1,268,268)) 
  subjects_adj = np.zeros((1,160,160)) 
  #25840
  for subject in range(start_value, end_value):
    subject_path = os.path.join(path,str(subject))
    
    if 'session_1' in os.listdir(subject_path):
    
       subjects_label = extract_data(subject,'session_1', 'shen_268', subjects_label)
       subjects_adj = extract_data(subject,'session_1', 'Dosenbach_160', subjects_adj)
       
#   for subject in range(25840,)
  return subjects_adj, subjects_label

def data():
  subjects_adj,subjects_labels = load_data(25629,25830)
  test_adj_1, test_labels_1 = load_data(25831, 25863)
  test_adj_2 , test_labels_2 = load_data(30701, 30757)
  test_adj = np.concatenate((test_adj_1,test_adj_2),axis=0)
  test_labels = np.concatenate((test_labels_1,test_labels_2),axis=0)
  return subjects_adj, subjects_labels, test_adj, test_labels