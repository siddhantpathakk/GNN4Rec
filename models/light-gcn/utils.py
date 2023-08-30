import torch
# load edges between users and movies
def load_edge_csv(df,
                  src_index_col,
                  dst_index_col,
                  link_index_col,
                  rating_threshold=3.5):
    edge_index = None
    src = [user_id for user_id in  df['userId']]

    num_users = len(df['userId'].unique())

    dst = [(movie_id) for movie_id in df['movieId']]

    link_vals = df[link_index_col].values

    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold

    edge_values = []

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
            edge_values.append(link_vals[i])

    # edge_values is the label we will use for compute training loss
    return edge_index, edge_values
