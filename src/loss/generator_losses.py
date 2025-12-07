import torch
import torch.nn.functional as F

from src.loss.base_loss import BaseLoss

class FeatureLoss(BaseLoss):
    def forward(self, fmaps_real, fmaps_gen, **kwargs):
        loss = 0
        for one_disc_fmaps_real, one_disc_fmaps_gen in zip(fmaps_real, fmaps_gen):
            for fmap_real, fmap_gen in zip(one_disc_fmaps_real, one_disc_fmaps_gen):
                loss += F.l1_loss(fmap_real, fmap_gen)
        return loss

class GeneratorLoss(BaseLoss):
    def forward(self, discs_gen_out, **kwargs):
        loss = 0
        for disc_gen_out in discs_gen_out:
            one_disc_loss = F.mse_loss(disc_gen_out, torch.ones_like(disc_gen_out))
            loss += one_disc_loss
        return loss

class L1MelLoss(BaseLoss):
    def forward(self, gen_mel, real_mel, **kwargs):
        return F.l1_loss(gen_mel, real_mel)
