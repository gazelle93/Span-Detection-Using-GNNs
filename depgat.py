import torch
import torch.nn as nn

class Dependency_GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, dependency_list):
        super(Dependency_GATLayer, self).__init__()
        # in_dim: number of tokens
        # out_dim: dimension of word embedding
        # dependency_list: the entire dependency types
        # reverse_case (default=True): Considering not only the result of dependency representation but also the reversed dependency representation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_weight = nn.Linear(out_dim*2, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def self_loop(self, _input, dependency_triples):
        self_loop_dict = {0:torch.zeros(self.out_dim)}
        h_dict = {0:torch.zeros(self.out_dim)}

        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]

            self_loop_dict[cur_dependent] = self.weight(_input[cur_governor].T)
            h_dict[cur_dependent] = self.weight(_input[cur_governor].T)

        return self_loop_dict, h_dict

    def self_attn_mechanism(self, _input, dependency_triples):
        e_tensor = torch.zeros(len(_input),len(_input))

        # egde attention
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]

            e_governor_dependent = self.attn_weight(torch.cat((self.weight(_input[cur_governor].T), self.weight(_input[cur_dependent].T)), -1))
            e_tensor[cur_governor, cur_dependent] = e_governor_dependent

        # Normalize edge attention
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]

            # masked attention
            zero_attn_mask = -1e18*torch.ones_like(e_tensor[cur_governor])
            masked_e = torch.where(e_tensor[cur_governor] > 0, e_tensor[cur_governor], zero_attn_mask)
            e_tensor[cur_governor] = self.softmax(masked_e.view(1,len(masked_e)))

        return e_tensor


    def forward(self, _input, dependency_triples):
        # self loop of each token
        self_loop_dict, h_dict = self.self_loop(_input, dependency_triples)

        # normalized attention score of each token
        attn_score_tensor = self.self_attn_mechanism(_input, dependency_triples)

        # Weighted sum based on the final attention weight
        for dep_triple in dependency_triples:
            cur_governor = dep_triple[2]
            cur_dependent = dep_triple[0]

            cur_attn = attn_score_tensor[cur_governor, cur_dependent] * self.weight(_input[cur_dependent].T)
            h_dict[cur_governor] += cur_attn

        output_list = list(h_dict.values())
        output_list = self.leakyrelu(torch.stack(output_list))

        return output_list

class Dependency_GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha,dependency_list, num_layers, dropout_rate):

        super(Dependency_GAT, self).__init__()
        # input_dim: number of tokens
        # hidden_dim: size of embeddings
        # output_dim: dimension of word embedding
        # dependency_list: the entire dependency types
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.num_layers = num_layers

        self.gat_layer = []
        for i in range(num_layers):
            self.gat_layer.append(Dependency_GATLayer(self.input_dim, self.hidden_dim, alpha, dependency_list))

        self.dropout = nn.Dropout(dropout_rate)
        self.ff_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)

    def forward(self, _input, dependency_triples):
        output = self.gat_layer[0](_input, dependency_triples)

        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                output = self.gat_layer[i+1](output, dependency_triples)

        output = self.dropout(output)

        output = self.ff_layer(output)

        return output
