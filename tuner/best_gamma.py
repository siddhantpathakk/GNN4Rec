from main import *
import json
print('---------------------')
print('Best gamma')
print('---------------------')
gammas = [0,1e-1,1e-2,1e-3,1e-4]
gamma_data={}

for i in gammas:
    print('Training with gamma: ',i)
    params['gamma'] = i
    
    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    
    ultragcn = ultragcn.to(params['device'])
    
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    
    params['max_epoch'] = 75

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)
    F1_score, Precision, Recall, NDCG = test(ultragcn, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])
    
    gamma_data[str(i)] = {'F1':F1_score,
                        'Precision':Precision,
                        'Recall':Recall,
                        'NDCG':NDCG}
    
    print('Training with gamma: ',i,' completed\n\n')


print(gamma_data)

with open('./logs/params/lr/gamma.json','w') as f:
    json.dump(gamma_data,f)
    print('Saved learning rate data at gamma.json')