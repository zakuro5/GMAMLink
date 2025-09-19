import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 d_k=32, d_r=0, d_c=128, d_c_prime=128,
                 dropout=0.1, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_r = d_r
        self.d_c = d_c
        self.d_c_prime = d_c_prime
        self.head_dim = embed_dim // num_heads

        # 查询投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 键/值低秩投影
        self.k_low_rank = nn.Linear(embed_dim, d_c, bias=bias)
        self.v_low_rank = nn.Linear(embed_dim, d_c, bias=bias)

        # 查询低秩投影
        self.q_low_rank = nn.Linear(embed_dim, d_c_prime, bias=bias)

        # 多头投影
        self.q_head_proj = nn.ModuleList([
            nn.Linear(d_c_prime, d_k, bias=bias) for _ in range(num_heads)
        ])

        self.k_head_proj = nn.ModuleList([
            nn.Linear(d_c, d_k, bias=bias) for _ in range(num_heads)
        ])

        self.v_head_proj = nn.ModuleList([
            nn.Linear(d_c, self.head_dim, bias=bias) for _ in range(num_heads)
        ])

        # RoPE投影 (可选)
        if d_r > 0:
            self.q_rope_proj = nn.ModuleList([
                nn.Linear(d_c_prime, d_r, bias=bias) for _ in range(num_heads)
            ])
            self.k_rope_proj = nn.Linear(embed_dim, d_r, bias=bias)
        else:
            self.q_rope_proj = None
            self.k_rope_proj = None

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale_factor = 1.0 / math.sqrt(d_k + d_r) if d_r > 0 else 1.0 / math.sqrt(d_k)

    def rope(self, x, positions):
        return x * torch.cos(positions) + x * torch.sin(positions)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, positions=None):
        tgt_len, batch_size, _ = query.shape
        src_len = key.size(0)

        q = self.q_proj(query)

        q_low = self.q_low_rank(q)
        k_low = self.k_low_rank(key)
        v_low = self.v_low_rank(value)

        if self.d_r > 0 and positions is not None:
            k_rope = self.k_rope_proj(key)
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(batch_size, -1)
            k_rope = self.apply_rope(k_rope, positions)

        # 多头处理
        attn_outputs = []
        attn_weights_list = [] if need_weights else None

        for head in range(self.num_heads):
            q_head = self.q_head_proj[head](q_low)
            k_head = self.k_head_proj[head](k_low)
            v_head = self.v_head_proj[head](v_low)

            if self.d_r > 0 and positions is not None:
                print(self.d_r)
                q_rope = self.q_rope_proj[head](q_low)
                q_rope = self.rope(q_rope, positions)
                q_head = torch.cat([q_head, q_rope], dim=-1)
                k_head = torch.cat([k_head, k_rope], dim=-1)

            # 调整形状用于注意力计算
            q_head = q_head.transpose(0, 1)
            k_head = k_head.transpose(0, 1)
            v_head = v_head.transpose(0, 1)

            # 计算注意力分数
            attn_scores = torch.bmm(q_head, k_head.transpose(1, 2)) * self.scale_factor

            # 应用掩码
            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(1).expand(-1, tgt_len, -1),
                    float('-inf')
                )

            # 计算注意力权重
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            if need_weights:
                attn_weights_list.append(attn_weights)

            # 加权求和
            head_output = torch.bmm(attn_weights, v_head)
            attn_outputs.append(head_output)

        attn_output = torch.cat(attn_outputs, dim=-1)
        attn_output = attn_output.transpose(0, 1)

        # 输出投影和dropout
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output