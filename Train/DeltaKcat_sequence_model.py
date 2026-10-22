"""Sequence-relative DeltaKcat model.

This file defines:
1) a residual projector for protein token embeddings,
2) a gated multi-head cross-attention block, and
3) the final pairwise regression model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Resnet_prot(nn.Module):
    """Two-layer residual MLP used to project protein token features."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x + F.relu(self.lin2(x))
        return x


class FC(nn.Module):
    """Fully connected regression head."""

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
        """Apply stacked MLP layers."""
        for layer in self.predict:
            h = layer(h)
        return h


class MultiHeadAtten_gate(nn.Module):
    """Cross-attention from molecule features to protein token features."""

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
        # Normalize both branches before attention.
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

        # Gate controls how much attended context is injected.
        combined = torch.cat([mol.unsqueeze(1), output], dim=-1)
        gate = torch.sigmoid(self.gate_linear(combined))
        output = output * gate

        if return_attention_weights:
            return self.dropout(output), attention_weights
        else:
            return self.dropout(output)


class DeltaKcat_gate(nn.Module):
    """Main model for sequence-relative delta prediction."""

    def __init__(self,
                 hidden_dim=512,
                 dropout=0.2,
                 device='cuda',
                ):
        super(DeltaKcat_gate, self).__init__()
        self.device = device
        self.mol_molt5 = nn.Linear(935, 512)
        self.res_prot1 = Resnet_prot(1024, 512)
        self.classify_feat = nn.Linear(hidden_dim, 256)
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.cross_att = MultiHeadAtten_gate(hidden_dim, 16, dropout)
        self.cross_att2 = MultiHeadAtten_gate(hidden_dim, 16, dropout)
        self.fc = FC(1536*3 + 768, hidden_dim, 2, dropout, 1)

    def forward(self, data, return_attention_weights=False):
        """Run pairwise forward pass and return predicted log2_delta_kcat."""

        data_feature = data['dlkcat_features']

        # Project token-level embeddings for sequence 1 and sequence 2.
        seq_emb_add_1 = self.res_prot1(data_feature.seq_emb_1)
        node_vector_1 = seq_emb_add_1.view(-1, 1024, 512)

        seq_emb_add_2 = self.res_prot1(data_feature.seq_emb_2)
        node_vector_2 = seq_emb_add_2.view(-1, 1024, 512)

        # Concatenate fingerprint features with MolT5 embedding, then project.
        mol = torch.cat([data_feature.smiles_emb, data_feature.smiles_emb_molt5], dim=1)
        mol = self.mol_molt5(mol)

        # Prepare protein token features for cross-attention.
        prot_1 = F.relu(self.linear(node_vector_1))
        prot_2 = F.relu(self.linear(node_vector_2))

        if return_attention_weights:
            # The same substrate is aligned with each candidate sequence.
            cross_mol_1, attn_weights_1 = self.cross_att(mol, prot_1, data_feature.mask_1,
                                                         return_attention_weights=True)
            cross_mol_2, attn_weights_2 = self.cross_att(mol, prot_2, data_feature.mask_2,
                                                         return_attention_weights=True)
        else:
            cross_mol_1 = self.cross_att(mol, prot_1, data_feature.mask_1)
            cross_mol_2 = self.cross_att(mol, prot_2, data_feature.mask_2)

        # Remove singleton sequence dimension from attention output.
        cross_mol_1 = cross_mol_1.squeeze(1)
        cross_mol_2 = cross_mol_2.squeeze(1)

        # Fuse mean sequence embeddings with cross-attended molecule context.
        cross_mol_1 = torch.cat((data_feature.mean_seq_1, cross_mol_1), dim=-1)
        cross_mol_2 = torch.cat((data_feature.mean_seq_2, cross_mol_2), dim=-1)

        # Final feature includes both branches and their difference.
        derta_input = torch.cat((data_feature.smiles_emb_molt5, cross_mol_1, cross_mol_2, (cross_mol_1 - cross_mol_2)),
                                dim=-1)
        kcat_pred = self.fc(derta_input).squeeze(dim=-1)
        if return_attention_weights:
            return kcat_pred, attn_weights_1, attn_weights_2
        else:
            return kcat_pred
