import torch
import numpy as np
import torch.nn.functional as F


###################################################################################################################
############################################ M E T R I C S ########################################################
###################################################################################################################

def dice_coef(groundtruth_masks, pred_masks, num_layers):
    dice_scores = []
    for layer in range(num_layers):
        intersect = torch.sum(pred_masks[:, layer, :, :] * groundtruth_masks[:, layer, :, :])
        total_sum = torch.sum(pred_masks[:, layer, :, :]) + torch.sum(groundtruth_masks[:, layer, :, :])
        dice = 2 * intersect / total_sum if total_sum > 0 else torch.tensor(0.0, device=groundtruth_masks.device)
        dice_scores.append(round(dice.item(), 3))
    return dice_scores





