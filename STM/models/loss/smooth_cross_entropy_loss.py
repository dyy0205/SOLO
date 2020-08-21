import torch
import torch.nn.functional as F


class SmoothCrossEntropyLoss(torch.nn.KLDivLoss):
    def __init__(self, eps=0.1, *args, **kargs):
        super(SmoothCrossEntropyLoss, self).__init__(*args, **kargs)
        self.eps = eps

    def forward(self, x, target):
        k = x.size(1)
        eps = self.eps / (k - 1)
        target_logit = torch.zeros(x.size(), device=x.device).fill_(eps)
        target_logit.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        if self.reduction=='mean':
            return super(SmoothCrossEntropyLoss, self).forward(
                F.log_softmax(x, 1), target_logit) * k
        elif self.reduction=='elementwise_mean':
            return super(SmoothCrossEntropyLoss, self).forward(
                F.log_softmax(x, 1), target_logit) * k
        elif self.reduction=='none':
            return super(SmoothCrossEntropyLoss, self).forward(
                F.log_softmax(x, 1), target_logit).sum(1)
        elif self.reduction=='sum':
            return super(SmoothCrossEntropyLoss, self).forward(
                F.log_softmax(x, 1), target_logit)
        else:
            raise Exception(f'No such reduction method: {self.reduction}')

class SmoothCrossEntropyLoss_2(torch.nn.KLDivLoss):
    def __init__(self, eps=0.1, *args, **kargs):
        super(SmoothCrossEntropyLoss_2, self).__init__(*args, **kargs)
        self.eps = eps

    def forward(self, x, target):
        # x is tensor after softmax
        k = x.size(1)
        eps = self.eps / (k - 1)
        target_logit = torch.zeros(x.size(), device=x.device).fill_(eps)
        target_logit.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        if self.reduction=='mean':
            return super(SmoothCrossEntropyLoss_2, self).forward(
                F.log_softmax(x, 1), target_logit) * k
        elif self.reduction=='elementwise_mean':
            return super(SmoothCrossEntropyLoss_2, self).forward(
                F.log_softmax(x, 1), target_logit) * k
        elif self.reduction=='none':
            return super(SmoothCrossEntropyLoss_2, self).forward(
                F.log_softmax(x, 1), target_logit).sum(1)
        elif self.reduction=='sum':
            return super(SmoothCrossEntropyLoss_2, self).forward(
                F.log_softmax(x, 1), target_logit)
        else:
            raise Exception(f'No such reduction method: {self.reduction}')


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ylabel = torch.tensor([[[1, 2]], [[3, 4]]], device=device)
    yprob = torch.tensor(
        [[[[0., 0.]],
          [[1., 0.]],
          [[0., 1.]],
          [[0., 0.]],
          [[0., 0.]]],
         [[[0., 0.]],
          [[0., 0.]],
          [[0., 0.]],
          [[1., 0.]],
          [[0., 1.]]]],
        device=device)

    x = torch.tensor(
        [[[[0., 0.]],
          [[1, 0.]],
          [[1, 1.]],
          [[0., 0.]],
          [[0., 0.]]],
         [[[0., 0.]],
          [[0., 0.3]],
          [[0., 0.]],
          [[1., 0.]],
          [[0., 2]]]],
        device=device)
    print(torch.nn.CrossEntropyLoss()(x, ylabel))
    print(F.cross_entropy(x, ylabel))
    xlogit = F.log_softmax(x, 1)
    print(torch.nn.functional.nll_loss(xlogit, ylabel))
    print(F.kl_div(xlogit, yprob) * 5)
    print(SmoothCrossEntropyLoss(eps=0)(x, ylabel))
    print('-----------')
    print(SmoothCrossEntropyLoss(eps=0.01)(x, ylabel))
    print(SmoothCrossEntropyLoss(eps=0.001)(x, ylabel))
    if torch.__version__ > '1.0.0':
        for mode in ['mean', 'none', 'sum']:
            assert (torch.nn.CrossEntropyLoss(reduction=mode)(x, ylabel).numpy()==SmoothCrossEntropyLoss(eps=0.0, reduction=mode)(x, ylabel).numpy()).all()
    else:
        for mode in ['elementwise_mean', 'none', 'sum']:
            assert (torch.nn.CrossEntropyLoss(reduction=mode)(x, ylabel).numpy()==SmoothCrossEntropyLoss(eps=0.0, reduction=mode)(x, ylabel).numpy()).all()
