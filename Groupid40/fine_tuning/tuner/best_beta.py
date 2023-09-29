from main import *
import json
from scipy.stats import loguniform
a = 1-1e-1
b= 1-1e-3
betas = loguniform.rvs(a, b, size=5)
betas_data={}
print('---------------------')
print('Best beta/momentum')
print('---------------------')

for i in betas:
    print('Training with beta: ',i)
    
    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    
    ultragcn = ultragcn.to(params['device'])
    
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'], betas=(i,0.99))
    
    params['max_epoch'] = 75
    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)
    F1_score, Precision, Recall, NDCG = test(ultragcn, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])