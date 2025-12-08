import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class Wav2Vec2MOS(nn.Module):
    sample_rate = 16_000

    def __init__(self, path, device, freeze=True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze

        self.dense = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        if self.freeze:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        state_dict = torch.load(path, map_location=device)["state_dict"]
        self.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
        self.eval()
        self.to(device)

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def forward(self, x):
        x = self.encoder(x)["last_hidden_state"]
        x = self.dense(x)
        return x.mean(dim=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
