import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import copy


class Resnet_prot(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x + F.relu(self.lin2(x))
        return x

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


class MultiHeadAtten_gate(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAtten_gate, self).__init__()
        # Define linear layers for q, k, v
        self.hidden_size = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.layer_norm_mol = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_prot = nn.LayerNorm(d_model, eps=1e-6)
        self.out_linear = nn.Linear(d_model, d_model)
        self.gate_linear = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mol, prot, mask=None, return_attention_weights=False):
        """
        Cross Attention with linear projections and masking.

        Parameters:
            q: Query tensor of shape (batch_size, 1, d_model)
            k: Key tensor of shape (batch_size, seq_len, d_model)
            v: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len)
                  where masked positions are `-inf` and unmasked are `0`.

        Returns:
            Output tensor of shape (batch_size, 1, d_model)
        """
        # Step 1: Linear transformation for q, k, v
        # mol = mol.view(-1, 1, self.hidden_size)
        # prot = prot.view(-1, 1024, self.hidden_size)
        batch_size = mol.size(0)
        mol = self.layer_norm_mol(mol)
        prot = self.layer_norm_prot(prot)
        q_proj = self.q_linear(mol).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, 1, d_model)
        k_proj = self.k_linear(prot).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, seq_len, d_model)
        v_proj = self.v_linear(prot).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, seq_len, d_model)

        attention_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.d_k ** 0.5)
        # Step 2: Calculate attention scores (batch_size, 1, seq_len)

        # Step 3: Apply mask (if provided)
        if mask is not None:
            mask = mask.view(-1,1,1024)
            mask = mask.unsqueeze(1)
            # mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))  # 无效位置设置为负无穷

        # Step 4: Apply softmax to get attention weights (batch_size, 1, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)



        # Step 5: Compute the weighted sum of values (batch_size, 1, d_model)
        output = torch.matmul(attention_weights, v_proj)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)  # (batch_size, 1, d_model)
        output = self.out_linear(output)

        combined = torch.cat([mol.unsqueeze(1), output], dim=-1)
        gate = torch.sigmoid(self.gate_linear(combined))
        output = output * gate

        if return_attention_weights:
            return self.dropout(output), attention_weights  # 注意力权重也返回
        else:
            return self.dropout(output)


class DelKcat_gate(nn.Module):
    def __init__(self,
                 mol_in_dim,
                 hidden_dim=128,
                 protein_dim=1280,
                 layer=10,
                 dropout=0.2,
                 att_layer=10,
                 device='cuda',
                 finetuning=False
                ):
        super(DelKcat_gate, self).__init__()
        self.device = device
        self.mol_molt5 = nn.Linear(935, 512)
        self.res_prot1 = Resnet_prot(1024, 512)
        self.classify_feat = nn.Linear(hidden_dim, 256)
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_att = MultiHeadAtten_gate(hidden_dim, 16, dropout)
        self.fc = FC(4864, hidden_dim, 2, dropout, 1)

    def forward(self, data, return_attention_weights=False):
        data_feature = data['dlkcat_features']
        seq_emb_add = self.res_prot1(data_feature.seq_emb)
        node_vector = seq_emb_add.view(-1, 1024, 512)

        mol_1 = torch.cat([data_feature.smiles_emb_1, data_feature.smiles_emb_molt5_1], dim=1)
        mol_1 = self.mol_molt5(mol_1)

        mol_2 = torch.cat([data_feature.smiles_emb_2, data_feature.smiles_emb_molt5_2], dim=1)
        mol_2 = self.mol_molt5(mol_2)

        prot_1 = F.relu(self.linear(node_vector))

        if return_attention_weights:
            cross_mol_1, attn_weights_1 = self.cross_att(mol_1, prot_1, data_feature.mask,
                                                         return_attention_weights=True)
            cross_mol_2, attn_weights_2 = self.cross_att(mol_2, prot_1, data_feature.mask,
                                                         return_attention_weights=True)
        else:
            cross_mol_1 = self.cross_att(mol_1, prot_1, data_feature.mask)
            cross_mol_2 = self.cross_att(mol_2, prot_1, data_feature.mask)

        cross_mol_1 = cross_mol_1.squeeze()
        cross_mol_2 = cross_mol_2.squeeze()
        cross_mol_1 = torch.cat((cross_mol_1, data_feature.smiles_emb_molt5_1), dim=-1)
        cross_mol_2 = torch.cat((cross_mol_2, data_feature.smiles_emb_molt5_2), dim=-1)
        derta_input = torch.cat((data_feature.mean_seq, cross_mol_1, cross_mol_2, (cross_mol_1 - cross_mol_2)), dim=-1)
        kcat_pred = self.fc(derta_input).squeeze(dim=-1)
        if return_attention_weights:
            return kcat_pred, attn_weights_1, attn_weights_2
        else:
            return kcat_pred