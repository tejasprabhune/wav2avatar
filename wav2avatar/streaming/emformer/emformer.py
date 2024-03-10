import torch
from torchaudio.models import Emformer

class EMAEmformer(torch.nn.Module):

    def __init__(
        self,
        input_dim=512,
        num_heads=8,
        ffn_dim=256,
        num_layers=15,
        segment_length=5,
        left_context_length=45,
        right_context_length=0
    ):
        super().__init__()

        self.emformer = Emformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            segment_length=segment_length,
            left_context_length=left_context_length,
            right_context_length=right_context_length
        )

        self.output_layer = torch.nn.Linear(512, 12)
    
    def forward(self, x, lengths=None):
        x, lengths = self.emformer(x, lengths)
        x = self.output_layer(x)
        x = x.transpose(2, 1)
        return x
    
    def infer(self, x, lengths, state):
        x, lengths, state = self.emformer.infer(x, lengths, state)
        x = self.output_layer(x)
        x = x.transpose(2, 1)
        return x, state