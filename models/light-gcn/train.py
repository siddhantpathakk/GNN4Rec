# import required modules
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import defaultdict

import torch
from torch import nn, optim, Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import download_url, extract_zip
from torch_sparse import SparseTensor

from utils import load_edge_csv

# defines LightGCN model
class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops


        # define user and item embedding for direct look up.
        # embedding dimension: num_user/num_item x embedding_dim
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0

        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0


        # "Fills the input Tensor with values drawn from the normal distribution"
        # according to LightGCN paper, this gives better performance
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        # create a linear layer (fully connected layer) so we can output a single value (predicted_rating)
        self.out = nn.Linear(embedding_dim + embedding_dim, 1)

    def forward(self, edge_index: Tensor, edge_values: Tensor):
        edge_index_norm = gcn_norm(edge_index=edge_index,
                                   add_self_loops=self.add_self_loops)

        # concat the user_emb and item_emb as the layer0 embing matrix
        # size will be (n_users + n_items) x emb_vector_len.   e.g: 10334 x 64
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0

        embs = [emb_0] # save the layer0 emb to the embs list

        # emb_k is the emb that we are actually going to push it through the graph layers
        # as described in lightGCN paper formula 7
        emb_k = emb_0

        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)


        # this is doing the formula8 in LightGCN paper

        # the stacked embs is a list of embedding matrix at each layer
        #    it's of shape n_nodes x (n_layers + 1) x emb_vector_len.
        #        e.g: torch.Size([10334, 4, 64])
        embs = torch.stack(embs, dim=1)

        # From LightGCn paper: "In our experiments, we find that setting α_k uniformly as 1/(K + 1)
        #    leads to good performance in general."
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(emb_final,
                                                       [self.num_users, self.num_items]) # splits into e_u^K and e_i^K


        r_mat_edge_index, _ = convert_adj_mat_edge_index_to_r_mat_edge_index(edge_index, edge_values)

        src, dest =  r_mat_edge_index[0], r_mat_edge_index[1]

        # applying embedding lookup to get embeddings for src nodes and dest nodes in the edge list
        user_embeds = users_emb_final[src]
        item_embeds = items_emb_final[dest]

        # output dim: edge_index_len x 128 (given 64 is the original emb_vector_len)
        output = torch.cat([user_embeds, item_embeds], dim=1)

        # push it through the linear layer
        output = self.out(output)

        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index, input_edge_values):
    R = torch.zeros((num_users, num_movies))
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = input_edge_values[i] # assign actual edge value to Interaction Matrix

    R_transpose = torch.transpose(R, 0, 1)

    # create adj_matrix
    adj_mat = torch.zeros((num_users + num_movies , num_users + num_movies))
    adj_mat[: num_users, num_users :] = R.clone()
    adj_mat[num_users :, : num_users] = R_transpose.clone()

    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo_indices = adj_mat_coo.indices()
    adj_mat_coo_values = adj_mat_coo.values()
    return adj_mat_coo_indices, adj_mat_coo_values

def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index, input_edge_values):

    sparse_input_edge_index = SparseTensor(row=input_edge_index[0],
                                           col=input_edge_index[1],
                                           value = input_edge_values,
                                           sparse_sizes=((num_users + num_movies), num_users + num_movies))

    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: num_users, num_users :]

    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    r_mat_edge_values = interact_mat.to_sparse_coo().values()

    return r_mat_edge_index, r_mat_edge_values

def get_recall_at_k(input_edge_index,
                     input_edge_values, # the true label of actual ratings for each user/item interaction
                     pred_ratings, # the list of predicted ratings
                     k=10,
                     threshold=3.5):
    with torch.no_grad():
        user_item_rating_list = defaultdict(list)

        for i in range(len(input_edge_index[0])):
            src = input_edge_index[0][i].item()
            dest = input_edge_index[1][i].item()
            true_rating = input_edge_values[i].item()
            pred_rating = pred_ratings[i].item()

            user_item_rating_list[src].append((pred_rating, true_rating))

        recalls = dict()
        precisions = dict()

        for user_id, user_ratings in user_item_rating_list.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        overall_recall = sum(rec for rec in recalls.values()) / len(recalls)
        overall_precision = sum(prec for prec in precisions.values()) / len(precisions)

        return overall_recall, overall_precision


if __name__ == "__main__":
    
    # download the dataset
    # https://grouplens.org/datasets/movielens/
    # "Small: 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users. Last updated 9/2018"
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    extract_zip(download_url(url, '.'), '.')

    movie_path = './ml-latest-small/movies.csv'
    rating_path = './ml-latest-small/ratings.csv'
    user_path = './ml-latest-small/users.csv'

    rating_df = pd.read_csv(rating_path)

    lbl_user = preprocessing.LabelEncoder()
    lbl_movie = preprocessing.LabelEncoder()

    rating_df.userId = lbl_user.fit_transform(rating_df.userId.values)
    rating_df.movieId = lbl_movie.fit_transform(rating_df.movieId.values)


    edge_index, edge_values = load_edge_csv(
        rating_df,
        src_index_col='userId',
        dst_index_col='movieId',
        link_index_col='rating',
        rating_threshold=1 # need to use threshold=1 so the model can learn based on RMSE
    )


    edge_index = torch.LongTensor(edge_index)
    edge_values = torch.tensor(edge_values)

    num_users = len(rating_df['userId'].unique())
    num_movies = len(rating_df['movieId'].unique())

    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(all_indices,
                                                test_size=0.2,
                                                random_state=1)

    val_indices, test_indices = train_test_split(test_indices,
                                                test_size=0.5,
                                                random_state=1)

    train_edge_index = edge_index[:, train_indices]
    train_edge_value = edge_values[train_indices]

    val_edge_index = edge_index[:, val_indices]
    val_edge_value = edge_values[val_indices]

    test_edge_index = edge_index[:, test_indices]
    test_edge_value = edge_values[test_indices]

    train_edge_index, train_edge_values  = convert_r_mat_edge_index_to_adj_mat_edge_index(train_edge_index, train_edge_value)
    val_edge_index, val_edge_values = convert_r_mat_edge_index_to_adj_mat_edge_index(val_edge_index, val_edge_value)
    test_edge_index, test_edge_values = convert_r_mat_edge_index_to_adj_mat_edge_index(test_edge_index, test_edge_value)

    # define contants
    ITERATIONS = 10000
    EPOCHS = 10

    BATCH_SIZE = 1024

    LR = 1e-3
    ITERS_PER_EVAL = 200
    ITERS_PER_LR_DECAY = 200
    K = 10
    LAMBDA = 1e-6
    
    # setup
    device = torch.device('cpu')
    print(f"Using device {device}.")

    layers = 1
    model = LightGCN(num_users=num_users,
                    num_items=num_movies,
                    K=layers)

    model = model.to(device)
    model.train()

    # add decay to avoid overfit
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)


    loss_func = nn.MSELoss()
    
    r_mat_train_edge_index, r_mat_train_edge_values = convert_adj_mat_edge_index_to_r_mat_edge_index(train_edge_index, train_edge_values)
    r_mat_val_edge_index, r_mat_val_edge_values = convert_adj_mat_edge_index_to_r_mat_edge_index(val_edge_index, val_edge_values)
    r_mat_test_edge_index, r_mat_test_edge_values = convert_adj_mat_edge_index_to_r_mat_edge_index(test_edge_index, test_edge_values)

    # training loop
    train_losses = []
    val_losses = []
    val_recall_at_ks = []
    
    for iter in tqdm(range(ITERATIONS)):
        # forward propagation

        # the rating here is based on r_mat
        pred_ratings = model.forward(train_edge_index, train_edge_values)


        train_loss = loss_func(pred_ratings, r_mat_train_edge_values.view(-1,1))


        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # going over validation set
        if iter % ITERS_PER_EVAL == 0:
            model.eval()

            with torch.no_grad():
                val_pred_ratings = model.forward(val_edge_index, val_edge_values)

                val_loss = loss_func(val_pred_ratings, r_mat_val_edge_values.view(-1,1)).sum()

                recall_at_k, precision_at_k = get_recall_at_k(r_mat_val_edge_index,
                                                            r_mat_val_edge_values,
                                                            val_pred_ratings,
                                                            k = 20
                                                            )


                val_recall_at_ks.append(round(recall_at_k, 5))
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())

                print(f"[Iteration {iter}/{ITERATIONS}], train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss.item(), 5)},  recall_at_k {round(recall_at_k, 5)}, precision_at_k {round(precision_at_k, 5)}")

            model.train()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()
            
    # plot the loss curves
    iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, val_losses, label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training and validation loss curves')
    plt.legend()
    plt.savefig('loss.png')


    f2 = plt.figure()
    plt.plot(iters, val_recall_at_ks, label='recall_at_k')
    plt.xlabel('iteration')
    plt.ylabel('recall_at_k')
    plt.title('recall_at_k curves')
    plt.savefig('recall_at_k.png')


    model.eval()
    with torch.no_grad():
        pred_ratings = model.forward(test_edge_index, test_edge_values)
        recall_at_k, precision_at_k = get_recall_at_k(r_mat_test_edge_index,
                                                    r_mat_test_edge_values,
                                                    pred_ratings, 20)
        print(f"recall_at_k {round(recall_at_k, 5)}, precision_at_k {round(precision_at_k, 5)}")