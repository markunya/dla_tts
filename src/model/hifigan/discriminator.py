import torch

from src.model.base_model import BaseModel
from src.model.hifigan.mpd import HifiMultiPeriodDiscriminator
from src.model.hifigan.msd import HifiMultiScaleDiscriminator

class HifiDiscriminator(BaseModel):
    def __init__(self):
        super().__init__()

        self.mpd = HifiMultiPeriodDiscriminator()
        self.msd = HifiMultiScaleDiscriminator()

    def forward(self, batch: dict) -> dict:
        real_wav = batch['wav']
        gen_wav = batch['gen_wav']

        mpd_out = self.mpd(real_wav, gen_wav)
        msd_out = self.msd(real_wav, gen_wav)

        return {
            'discs_real_out': [*mpd_out[0], *msd_out[0]],
            'discs_gen_out': [*mpd_out[1], *msd_out[1]],
            'fmaps_real': [*mpd_out[2], *msd_out[2]],
            'fmaps_gen': [*mpd_out[3], *msd_out[3]]
        }