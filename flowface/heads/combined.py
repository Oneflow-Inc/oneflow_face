from random import sample
import oneflow as flow
from flowface.heads.common import FC


class Combined(flow.nn.Module):
    def __init__(self, scale, m1, m2, m3):
        super(Combined, self).__init__()
        self.scale = scale
        self.margin_loss = flow.nn.CombinedMarginLoss(m1=m1, m2=m2, m3=m3)

    def forward(self, logits: flow.Tensor, labels: flow.Tensor):
        return self.margin_loss(logits, labels) * self.scale

        
class CombinedFC(flow.nn.Module):
    def __init__(self, num_classes, embedding_size, is_global, is_parallel, sample_rate, *, m1, m2, m3, scale=64) -> None:
        super().__init__()
        self.fc = FC(embedding_size, num_classes, is_global=is_global, is_parallel=is_parallel, sample_rate=sample_rate)
        self.head = Combined(scale, m1, m2, m3)
        self.loss = flow.nn.functional.sparse_softmax_cross_entropy
        self.weight = self.fc.weight

    def forward(self, features, labels):
        logits, labels = self.fc(features, labels)
        logits = self.head(logits, labels)
        # the parameter order is [labels, logits]
        loss = self.loss(labels, logits).mean()
        return loss

if __name__ == "__main__":
    fc = CombinedFC(100, 128, is_global=False, is_parallel=True, sample_rate=1, m1=1, m2=0.5, m3=0)
    features = flow.randn(4, 128).requires_grad_()
    labels = flow.randint(0, 100, (4, ))
    fc(features, labels).backward()

