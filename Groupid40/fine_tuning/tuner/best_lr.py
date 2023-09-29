from main import *
import json
print('---------------------')
print('Best learning rate')
print('---------------------')
lrs = [1e-1,1e-2, 1e-3,1e-4]
lrs_data={}

for i in lrs:
    print('Training with lr: ',i)
    params['lr'] = i
    
    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    
    ultragcn = ultragcn.to(params['device'])
    
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    
    params['max_epoch'] = 75
    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)
    F1_score, Precision, Recall, NDCG = test(ultragcn, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])
    
    lrs_data[str(i)] = {'F1':F1_score,
                        'Precision':Precision,
                        'Recall':Recall,
                        'NDCG':NDCG}
    
    print('Training with lr: ',i,' completed\n\n')


print(lrs_data)

with open('./logs/opt/lr/learning_rate.json','w') as f:
    json.dump(lrs_data,f)
    print('Saved learning rate data at learning_rate.json')