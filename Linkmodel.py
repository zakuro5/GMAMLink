from GCN import *
from MultiheadAttention import *

# GCN + attention
class LinkModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, origin_dim, origin_output_dim):
        super(LinkModel, self).__init__()

        self.tf_linear = nn.Linear(input_dim, hidden_dim)
        self.target_linear = nn.Linear(input_dim, hidden_dim)

        self.attention = MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            d_k=32,
            d_r=0,  # 禁用RoPE
            d_c=64,
            d_c_prime=64
        )

        self.tfa_linear = nn.Linear(hidden_dim, output_dim)
        self.targeta_linear = nn.Linear(hidden_dim, output_dim)
        self.gcn = GCNConv(origin_dim, origin_output_dim)
        self.linear = nn.Linear(origin_output_dim, hidden_dim)

    def forward(self, x, train_sample, data_feature, adj):
        # GCN编码
        x_g = self.gcn(data_feature, adj)
        #       x_g = torch.sigmoid(x_g)
        x_g = F.leaky_relu(self.linear(x_g))
        x_g = torch.sigmod(x_g)

        tf_embed = F.leaky_relu(self.tf_linear(x))
        target_embed = F.leaky_relu(self.target_linear(x))

        # 调整 [num_nodes, hidden_dim] -> [num_nodes, 1, hidden_dim]
        x_g.size()

        x_g = x_g.unsqueeze(1)
        tf_embed = tf_embed.unsqueeze(1)
        target_embed = target_embed.unsqueeze(1)

        x_f = self.attention(
            query=x_g,
            key=tf_embed,
            value=target_embed
        )

        x_f = x_f.squeeze(1)

        tfa_embed = F.leaky_relu(self.tfa_linear(x_f))
        targeta_embed = F.leaky_relu(self.targeta_linear(x_f))

        train_tf = tfa_embed[train_sample[:, 0]]
        train_target = targeta_embed[train_sample[:, 1]]

        pred = torch.mul(train_tf, train_target)
        pred = torch.sum(pred, dim=1).view(-1, 1)

        #     pred = torch.cosine_similarity(train_tf,train_target,dim=1).view(-1,1)

        return pred