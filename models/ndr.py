"""
This code is adapted from the 'https://github.com/robertcsordas/ndr' repository.
"""

def sinusoidal_pos_embedding(d_model: int, max_len: int = 5000, pos_offset: int = 0,
                             device: Optional[torch.device] = None):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1) + pos_offset
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
        batch_first: if true, batch dimension is the first, if not, its the 2nd.
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False,
                 scale: float = 1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = sinusoidal_pos_embedding(d_model, max_len, 0) * scale

        self.batch_dim = 0 if batch_first else 1
        pe = pe.unsqueeze(self.batch_dim)

        self.register_buffer('pe', pe)

    def get(self, n: int, offset: int) -> torch.Tensor:
        return self.pe.narrow(1 - self.batch_dim, start=offset, length=n)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.get(x.size(1 - self.batch_dim), offset)
        return self.dropout(x)


class UniversalTransformerRandomLayerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int , n_extra: int = 0, n_test: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.n_extra = n_extra
        self.n_layers = n_layers
        self.n_test = n_test

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def forward(self, data: torch.Tensor, *args, **kwargs):
        self.set_n_layers(np.random.randint(self.n_layers, self.n_extra + self.n_layers + 1) if self.training else \
                          (self.n_test or self.n_layers))
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data



def UniversalTransformerRandomLayerEncoderWithLayer(layer):
    return lambda *args, **kwargs: UniversalTransformerRandomLayerEncoder(layer, *args, **kwargs)


class NDREncoder(torch.nn.Module):
        
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                dim_feedforward: int = 2048, dropout: float = 0.1,
                activation: ActivationFunction = F.relu, encoder_layer=UniversalTransformerRandomLayerEncoderWithLayer,
                attention_dropout: float = 0, attention_type: str = 'regular', n_input_tokens: int = 300, p_gate_drop: float = 0.05, pos_encod: bool = True, embedding_init: str = 'uniform'):

        super().__init__()

        self.embedding = torch.nn.Embedding(n_input_tokens + 3, d_model)
        self.pos_encod = pos_encod
        self.embedding_init = embedding_init
        # self.pos = framework.layers.PositionalEncoding(d_model, max_len=100, batch_first=True,
        #                                 scale= 1.0)
        if self.pos_encod:
            self.pos = framework.layers.PositionalEncoding(d_model, max_len=100, batch_first=True, scale= 1.0)
        else:
            self.pos = (lambda x, offset: x) 

        self.register_buffer('int_seq', torch.arange(100, dtype=torch.long))

        self.encoder = encoder_layer(layer = NDRGeometric if attention_type == 'geometric' else NDRResidual)(n_layers = num_encoder_layers, d_model = d_model, nhead = nhead, dim_feedforward = dim_feedforward,
                                    dropout = dropout, activation = activation, attention_dropout = attention_dropout, p_gate_drop = p_gate_drop)

        self.reset_parameters()

    def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
        return self.int_seq[: max_len] >= len.unsqueeze(-1)

    def forward(self, input_ids: torch.Tensor, src_len: torch.Tensor, attention_mask: Optional[AttentionMask] = None):

        src = self.pos(self.embedding(input_ids.long()), 0)
        in_len_mask = self.generate_len_mask(src.shape[1], src_len)
        memory = self.encoder(src, AttentionMask(in_len_mask, None))
        return memory

    def reset_parameters(self):

        if self.embedding_init == "xavier":
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif self.embedding_init == "kaiming":
            torch.nn.init.kaiming_normal_(self.embedding.weight)

        if self.output_map == "linear":
            torch.nn.init.xavier_uniform_(self.output_map.weight)