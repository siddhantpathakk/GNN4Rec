import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
	neg_candidates = np.arange(item_num)

	if sampling_sift_pos:
		neg_items = []
		for u in pos_train_data[0]:
			probs = np.ones(item_num)
			probs[interacted_items[u]] = 0
			probs /= np.sum(probs)

			u_neg_items = np.random.choice(neg_candidates, size = neg_ratio, p = probs, replace = True).reshape(1, -1)
	
			neg_items.append(u_neg_items)

		neg_items = np.concatenate(neg_items, axis = 0) 
	else:
		neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace = True)
	
	neg_items = torch.from_numpy(neg_items)
	
	return pos_train_data[0], pos_train_data[1], neg_items	# users, pos_items, neg_items



class UltraGCN(nn.Module):
    def __init__(self, params, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']

        self.negative_weight = params['negative_weight']
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.initial_weight = params['initial_weight']
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
        
        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.flatten()]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight
      
        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users).unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)
        
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
         
        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.user_embeds.weight.device
