import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
from torch.autograd import Variable
import time, random
from einops import rearrange


class FocalLoss(Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        input
            - sentence_length x vocab size
        target
            - sentence_length x vocab size
        """
        # target = target.view(-1,1)
        target = rearrange(target, pattern="sentence_length -> sentence_length 1")
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = rearrange(logpt, pattern="sentence_length 1 -> sentence_length")
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# unit test
if __name__ == "__main__":
    start_time = time.time()
    maxe = 0
    for i in range(1000):
        x = torch.rand(12800, 2) * random.randint(1, 10)
        x = Variable(x.cuda())
        l = torch.rand(12800).ge(0.1).long()
        l = Variable(l.cuda())

        output0 = FocalLoss(gamma=2)(x, l)
        output1 = nn.CrossEntropyLoss()(x, l)
        a = output0.item()
        b = output1.item()
        if abs(a - b) > maxe:
            maxe = abs(a - b)
    print("time:", time.time() - start_time, "max_error:", maxe)

    start_time = time.time()
    maxe = 0
    for i in range(100):
        x = torch.rand(128, 1000, 8, 4) * random.randint(1, 10)
        x = Variable(x.cuda())
        l = torch.rand(128, 8, 4) * 1000  # 1000 is classes_num
        l = l.long()
        l = Variable(l.cuda())

        output0 = FocalLoss(gamma=0)(x, l)
        output1 = nn.NLLLoss2d()(F.log_softmax(x), l)
        a = output0.data[0]
        b = output1.data[0]
        if abs(a - b) > maxe:
            maxe = abs(a - b)
    print("time:", time.time() - start_time, "max_error:", maxe)
