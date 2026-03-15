from GCN import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import sce_loss,sig_loss
from functools import partial

class GraphMAE(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_hidden, device, batchnorm=True):
        super(GraphMAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        enc_in_dim = input_dim
        enc_num_hidden = num_hidden
        enc_out_dim = num_hidden

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden

        self.encoder1 = GCNConv(input_dim, enc_num_hidden)
        self.encoder2 = GCNConv(enc_num_hidden, enc_out_dim)

        self.decoder = GCNConv(dec_in_dim, input_dim)

        #       self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers,dec_in_dim)

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))

        self.tf_linear = nn.Linear(num_hidden, output_dim)

        self.target_linear = nn.Linear(num_hidden, output_dim)

        self.MLP = nn.Linear(2 * output_dim, 2)

        self.criterion = self.setup_loss_fn()

    def encoding_mask_noise(self, x, adj, mask_rate=0.25):
        num_nodes = x.size()[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes);

    def encode(self, x, adj):
        # encoder forward
        x1 = self.encoder1(x, adj)

        x2 = self.encoder2(x1, adj)

        return x2

    def decode(self, x, adj):
        return self.decoder(x, adj)

    def setup_loss_fn(self, loss_fn='mse'):
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def _attr_prediction(self, x, adj):
        # forward
        # mask
        u_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, adj)
        # encode
        enc_rep = self.encode(u_x, adj)
        # attribute reconstruction
        rep = self.encoder_to_decoder(enc_rep)
        # if mlp/linear decoder -> pred ...
        # GNNdecoder re-mask
        rep[mask_nodes] = 0.0
        recon = self.decoder(rep, adj)

        # 特征重构损失
        x_t = x[mask_nodes]
        x_p = recon[mask_nodes]
        loss = self.criterion(x_t, x_p)

        return loss

    def forward(self, x, adj):
        loss = self._attr_prediction(x, adj);
        loss_item = {'loss': loss.item()}

        return loss, loss_item

    def get_embed(self, x, adj):
        return self.decode(x, adj)
