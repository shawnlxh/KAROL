from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn.functional import normalize
import torch
from torch import Tensor

class SSL_block(nn.Module):
    def __init__(self, hidden_channels, tau_u = 1, tau_l = 0.05):
        super(SSL_block, self).__init__()
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
        self.tau_u = tau_u
        self.tau_l = tau_l
    
    def forward(self,x1: Tensor, x2: Tensor, x1_index: Tensor, x2_index:Tensor):
        x1_feat = F.embedding(x1_index,x1)
        #print('x1_feat',x1_feat.size())
        x2_feat = F.embedding(x2_index,x2)
        x1_feat = F.normalize(x1_feat, dim=1)
        x2_feat = F.normalize(x2_feat, dim=1)
        sim_mt = torch.mul(x1_feat, x2_feat)
        sim_mt = self.lin1(sim_mt)
        sim_mt = F.leaky_relu(sim_mt, 0.2)
        sim_mt = F.sigmoid(self.lin2(sim_mt))
        ssl_temp = (self.tau_u - self.tau_l) * (1-sim_mt) + self.tau_l
        pos_rating = torch.sum(torch.mul(x1_feat,x2_feat),dim = -1)
        pos_rating = torch.exp(pos_rating/ssl_temp)
        tot_rating = torch.matmul(x1_feat, torch.transpose(x2_feat, 0, 1))
        tot_rating = torch.exp(tot_rating/ssl_temp)
        tot_rating = torch.sum(tot_rating, dim = 1)
        ssl_temp_avg = torch.mean(ssl_temp)
        return pos_rating, tot_rating, ssl_temp_avg
    
    
    
class MMCL_INV(nn.Module):
    def __init__(self, sigma=0.07, contrast_mode='all',
                 base_sigma=0.07, batch_size=256, anchor_count=2, C=1.0, kernel='rbf',reg=0.1, schedule=[], multiplier=2):
        super(MMCL_INV, self).__init__()
        self.sigma = sigma
        self.contrast_mode = contrast_mode
        self.base_sigma = base_sigma
        self.C = C
        self.kernel = kernel
        
        nn = batch_size - 1
        bs = batch_size
        
        self.mask, self.logits_mask = self.get_mask(batch_size, anchor_count)
        self.eye = torch.eye(anchor_count*batch_size).cuda()
        
        self.pos_mask = self.mask[:bs, bs:].bool()
        neg_mask=(self.mask*self.logits_mask+1)%2; 
        self.neg_mask = neg_mask-self.eye
        self.neg_mask = self.neg_mask[:bs, bs:].bool()
        
        self.kmask = torch.ones(batch_size,).bool().cuda()
        self.kmask.requires_grad = False
        self.reg = reg

        self.oneone = (torch.ones(bs, bs) + torch.eye(bs)*reg).cuda()
        self.one_bs = torch.ones(batch_size, nn, 1).cuda()
        self.one = torch.ones(nn,).cuda()
        self.KMASK = self.get_kmask(bs)
        self.block = torch.zeros(bs,2*bs).bool().cuda(); self.block[:bs,:bs] = True
        self.block12 = torch.zeros(bs,2*bs).bool().cuda(); self.block12[:bs,bs:] = True
        self.no_diag = (1-torch.eye(bs)).bool().cuda()
        self.bs = bs
        self.schedule = schedule
        self.multiplier = multiplier

    def get_kmask(self, bs):
        KMASK = torch.ones(bs, bs, bs).bool().cuda()
        for t in range(bs):
            KMASK[t,t,:] = False
            KMASK[t,:,t] = False
        return KMASK.detach()
        
    def get_mask(self, batch_size, anchor_count):
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        return mask, logits_mask
        
    def forward(self, x1: Tensor, x2: Tensor, x1_index: Tensor, x2_index:Tensor):
        
        x1_feat = F.embedding(x1_index,x1)
        #print('x1_feat',x1_feat.size())
        x2_feat = F.embedding(x2_index,x2)
        x1_feat = F.normalize(x1_feat, dim=1)
        x2_feat = F.normalize(x2_feat, dim=1)
        z = torch.concat((x1_feat,x2_feat),0)
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = F.normalize(z, p=2, dim=1)


        bs = ftr.shape[0]//2
        nn = bs - 1
        K = compute_kernel(ftr[:nn+1], ftr, kernel_type=self.kernel, gamma=self.sigma)
        
        with torch.no_grad():
            KK = torch.masked_select(K.detach(), self.block).reshape(bs, bs)
        
            KK_d0 = KK*self.no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1,bs,1)
            KXY = KXY + KXY.transpose(2,1)
            Delta = (self.oneone + KK).unsqueeze(0) + KXY
    
            DD = torch.masked_select(Delta, self.KMASK).reshape(bs, nn, nn)
            
            alpha_y, _ = torch.solve(2*self.one_bs, DD)

            # if torch.rand(1)>0.99: 
            #     print('alpha_y.max=%f alpha_y.min=%f alpha_y.mean=%f: error=%f'%
            #           (alpha_y.max(), alpha_y.min(),alpha_y.mean(), (torch.bmm(DD, alpha_y)-2.*self.one_bs).norm()))
                
            alpha_y = alpha_y.squeeze(2)
            if self.C == -1:
                alpha_y = torch.relu(alpha_y)
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=self.C).detach()
            alpha_x = alpha_y.sum(1)
            
        Ks = torch.masked_select(K, self.block12).reshape(bs, bs)
        Kn = torch.masked_select(Ks.T, self.neg_mask).reshape(bs,nn).T
        
        pos_loss = (alpha_x*(Ks*self.pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T*Kn).sum()/bs
        loss = neg_loss - pos_loss

        num_zero = (alpha_y == 0).sum()/alpha_y.numel()
        sparsity = (alpha_y == self.C).sum()/((alpha_y>0).sum()+1e-10)
        return loss

    
def bpr_loss(positives, negatives,lambda_reg=0):
    """
    users: tensor of user embeddings
    positive_items: tensor of positive item embeddings
    negative_items: tensor of negative item embeddings
    """
    # Calculate the predicted scores for both positive and negative items
    #pos_scores = torch.sum(users * positive_items, dim=-1)
    #neg_scores = torch.sum(users * negative_items, dim=-1)
    # Calculate the difference between the positive and negative scores
    #log_prob = F.logsigmoid(positives - negatives).mean()
    #log_prob = -torch.mean(F.logsigmoid(positives-negatives))
    log_prob = F.softplus(-(positives-negatives),beta=1).mean()
    regularization = 0
    if lambda_reg != 0:
        regularization = lambda_reg * parameters.norm(p=2).pow(2)
        regularization = regularization / positives.size(0)

    return log_prob + regularization