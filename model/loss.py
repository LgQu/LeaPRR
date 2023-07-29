import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation


    def forward(self, sim, targets):
        num_i, num_t = sim.size()
        loss = 0.
        for i in range(num_i):
            posi_sim = torch.masked_select(sim[i], targets[i])
            min_posi_sim = posi_sim.min()
            cost = (self.margin + sim[i] - min_posi_sim).clamp(min=0)
            cost = cost.masked_fill_(targets[i], 0)
            if self.max_violation:
                cost = cost.max()
            cost = cost.sum()
            loss += cost
        
        loss = loss / num_i
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, ):
        super(ContrastiveLoss, self).__init__()

    def forward(self, exp_sim, targets):
        loss = torch.mean(-1.0 * torch.log(torch.sum(exp_sim * targets, dim=-1) / torch.sum(exp_sim, dim=-1)))

        return loss
