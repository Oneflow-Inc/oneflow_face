import oneflow as flow

from flowface.heads.common import FC


class ArcFace(flow.nn.Module):
    def __init__(self, scale=64, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = scale
        self.margin_loss = flow.nn.CombinedMarginLoss(m1=1, m2=margin, m3=0)

    def forward(self, logits: flow.Tensor, labels: flow.Tensor):
        return self.margin_loss(logits, labels) * self.scale


class ArcFaceFC(flow.nn.Module):
    def __init__(
        self, num_classes, embedding_size, is_global, is_parallel, sample_rate, scale=64, margin=0.5
    ) -> None:
        super().__init__()
        self.fc = FC(
            embedding_size,
            num_classes,
            is_global=is_global,
            is_parallel=is_parallel,
            sample_rate=sample_rate,
        )
        self.head = ArcFace(scale, margin)
        self.loss = flow.nn.functional.sparse_softmax_cross_entropy
        self.weight = self.fc.weight

    def forward(self, features, labels):
        logits, labels = self.fc(features, labels)
        logits = self.head(logits, labels)
        # the parameter order is [labels, logits]
        loss = self.loss(labels, logits).mean()
        return loss


if __name__ == "__main__":
    fc = ArcFaceFC(128, 100, 64, 0.5)
    features = flow.randn(4, 128).requires_grad_()
    labels = flow.randint(0, 100, (4,))
    fc(features, labels).backward()
