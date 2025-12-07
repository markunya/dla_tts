from src.loss.base_loss import BaseLoss
from typing import List

class LossBuilder:
    def __init__(self, losses: List[BaseLoss]):
        self.losses = losses

    def calculate_loss(self, **kwargs):
        loss_dict = {}
        total_loss = 0.0

        for loss in self.losses:
            loss_value = loss(**kwargs)
            loss_dict[loss.tag] = loss_value.item()
            total_loss += loss.coef * loss_value

        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
