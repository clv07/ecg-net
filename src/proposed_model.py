import torch
import torch.nn as nn

class PerLeadCNN(nn.Module):
    """
    Grouped Conv1d with groups=num_leads.
    Output: (B, num_leads, lead_dim, T')
    """
    def __init__(self, num_leads=12, lead_dim=32):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim  = lead_dim

        c1, c2 = 8, 16
        self.cnn = nn.Sequential(
            # input  : (B, 12, 1000)
            # output : (B, 12 * c1, 500)
            nn.Conv1d(num_leads, num_leads * c1, kernel_size=15,
                      stride=2, padding=7, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                               

            nn.Conv1d(num_leads * c1, num_leads * c2, kernel_size=7,
                      stride=2, padding=3, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(num_leads * c2, num_leads * lead_dim, kernel_size=3,
                      stride=1, padding=1, groups=num_leads, bias=False),
            nn.BatchNorm1d(num_leads * lead_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, num_leads, T)
        B, L, T = x.shape
        h = self.cnn(x)                                    # (B, L*lead_dim, T')
        T_prime = h.size(-1)
        # reshape to expose lead dimension
        h = h.view(B, L, self.lead_dim, T_prime)           # (B, 12, lead_dim, T')
        return h

class SpatialLeadAttention(nn.Module):
    """
    Apply multi-head self-attention across the 12 leads
    at each downsampled time step independently to learn cross-lead relationships.

    Input:  (B, 12, lead_dim, T')
    Output: (B, 12, lead_dim, T')
    """
    def __init__(self, num_leads=12, lead_dim=32, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.lead_pos_embed = nn.Parameter(torch.randn(1, num_leads, lead_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lead_dim,
            nhead=nhead,
            dim_feedforward=lead_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.num_leads = num_leads
        self.lead_dim  = lead_dim

    def forward(self, x):
        B, L, D, T_prime = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * T_prime, L, D)
        x = x + self.lead_pos_embed
        x = self.attn(x)
        x = x.view(B, T_prime, L, D).permute(0, 2, 3, 1).contiguous()
        return x


class CnnTransformer(nn.Module):
    def __init__(self, num_leads=12, num_classes=5, lead_dim=32, d_model=192, nhead=4, num_layers=3,
                 dim_feedforward=384, lead_nhead=4, lead_layers=1, dropout=0.15):
        super().__init__()
        self.num_leads = num_leads
        self.lead_dim = lead_dim
        self.d_model = d_model

        # per-lead CNN
        self.per_lead_cnn = PerLeadCNN(num_leads=num_leads, lead_dim=lead_dim)

        # spatial lead attention - cross-lead at each timestep
        self.lead_attention = SpatialLeadAttention(
            num_leads=num_leads,
            lead_dim=lead_dim,
            nhead=lead_nhead,
            num_layers=lead_layers,
            dropout=dropout,
        )

        # merge leads: project (12 * lead_dim) -> d_model at each timestep
        self.lead_merge = nn.Sequential(
            nn.Linear(num_leads * lead_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # temporal transformer
        self.time_pos_embed = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_norm = nn.LayerNorm(d_model)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.per_lead_cnn(x)
        h = self.lead_attention(h)
        B, L, D, T_prime = h.shape
        h = h.permute(0, 3, 1, 2).contiguous().view(B, T_prime, L * D)
        h = self.lead_merge(h)                           

        h = h + self.time_pos_embed[:, :T_prime, :]
        h = self.temporal_transformer(h)
        h = self.temporal_norm(h)

        h = h.mean(dim=1)
        return self.classifier(h)
