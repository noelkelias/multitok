def create_pos_emb_matrix(dict_values, emb_length):
  emb_mat = {}
  for val in dict_values:
    # emb_mat[val] = list(np.random.uniform(low=-1, high=1, size=(emb_length,)))

    tokens = val.split()
    if len(tokens) == 1:
      emb_mat[val] = list(np.random.uniform(low=-1, high=1, size=(emb_length,)))
      # emb_mat[val] = random.sample(range(1, 5), emb_length)

    else:
      scale_factor = 1
      embedding = [0] * emb_length
      for tok in tokens:
        embedding = list(map(add, embedding, [x * scale_factor for x in emb_mat[tok]]))
        scale_factor = scale_factor*10
        embedding = list(map(add, embedding, emb_mat[tok]))

      emb_mat[val] = embedding
  return torch.FloatTensor(list(emb_mat.values()))
