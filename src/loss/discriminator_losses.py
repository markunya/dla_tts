import torch
import torch.nn.functional as F

from src.loss.base_loss import BaseLoss

class DiscriminatorLoss(BaseLoss):        
    def forward(self, discs_real_out, discs_gen_out, **kwargs):
        loss = 0
        for disc_real_out, disc_gen_out in zip(discs_real_out, discs_gen_out):
            real_one_disc_loss = F.mse_loss(disc_real_out, torch.ones_like(disc_real_out))
            gen_one_disc_loss = F.mse_loss(disc_gen_out, torch.zeros_like(disc_gen_out))
            loss += (real_one_disc_loss + gen_one_disc_loss)
        return loss
