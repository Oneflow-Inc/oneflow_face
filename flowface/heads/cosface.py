import oneflow as flow

from flowface.heads.common import FC


class CosFace(flow.nn.Module):
    def __init__(self, scale=64, margin=0.5):
        super(CosFace, self).__init__()
        self.scale = scale
        self.margin_loss = flow.nn.CombinedMarginLoss(m1=1, m2=0, m3=margin)

    def forward(self, logits: flow.Tensor, labels: flow.Tensor):
        return self.margin_loss(logits, labels) * self.scale


class CosFaceFC(flow.nn.Module):
    def __init__(self, num_classes, embedding_size, scale=64, margin=0.5) -> None:
        super().__init__()
        self.fc = FC(embedding_size, num_classes)
        self.head = CosFace(scale, margin)
        self.loss = flow.nn.functional.sparse_softmax_cross_entropy
        self.weight = self.fc.weight

    def forward(self, features, labels):
        logits, labels = self.fc(features, labels)
        logits = self.head(logits, labels)
        loss = self.loss(labels, logits)
        return loss


if __name__ == "__main__":
    fc = CosFaceFC(128, 100, 64, 0.5)
    features = flow.randn(4, 128).requires_grad_()
    labels = flow.randint(0, 100, (4,))
    fc(features, labels).backward()
