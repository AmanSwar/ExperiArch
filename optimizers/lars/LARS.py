import torch
from torch.optim import Optimizer

class LARS(Optimizer):

    def __init__(self , params , lr , momentum=0.9 , weight_decay=0.0005 , trust_coeff=0.001 , eps=1e-8):

        default = dict(lr=lr , momentum=momentum , weight_decay=weight_decay , trust_coeff=trust_coeff , eps=eps)
        super(LARS, self).__init__(params , default)

    @torch.no_grad()
    def step(self , closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        
        for grp in self.param_groups:
            for param in grp['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                weight_norm = torch.norm(param)
                grad_norm = torch.norm(grad)

                if grp['weight_decay'] != 0:
                    grad = grad.add(param , alpha=grp['weight_decay'])
                
                if weight_norm > 0 and grad_norm > 0:
                    adaptive_lr = grp['trust_coeff'] * (weight_norm / (grad_norm + grp['eps']))

                else:
                    adaptive_lr =1

                if 'momentum_buffer' not in self.state[param]:
                    buf = self.state[param]['momentum_buffer'] = torch.clone(grad).detach()

                else:
                    buf = self.state[param]['momentum_buffer']
                    buf.mul_(grp['momentum']).add_(grad , alpha=adaptive_lr)

                param.add_(buf , alpha=-grp['lr'])

        return loss
    

