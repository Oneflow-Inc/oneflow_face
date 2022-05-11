import math
import oneflow as flow

class FC7(flow.nn.Module):
    def __init__(self, embedding_size, num_classes, cfg, partial_fc=False, bias=False):
        super(FC7, self).__init__()
        self.weight = flow.nn.Parameter(flow.empty(num_classes, embedding_size))
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)

        self.partial_fc = partial_fc

        sample_rate = 1
        size = flow.env.get_world_size()
        num_local = (num_classes + size - 1) // size
        self.num_sample = int(num_local * sample_rate)
        self.total_num_sample = self.num_sample * size

    def forward(self, x, label):
        x = flow.nn.functional.normalize(x, dim=1)
        if self.partial_fc:
            (mapped_label, sampled_label, sampled_weight,) = flow.distributed_partial_fc_sample(
                weight=self.weight,
                label=label,
                num_sample=self.total_num_sample,
            )
            label = mapped_label
            weight = sampled_weight
        else:
            weight = self.weight
        weight = flow.nn.functional.normalize(weight, dim=1)
        x = flow.matmul(x, weight, transpose_b=True)

        return x, label

class ArcFace(flow.nn.Module):
    def __init__(self, scale=64, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = scale
        self.margin_loss = flow.nn.CombinedMarginLoss(m1=1, m2=margin, m3=0)

    def forward(self, logits: flow.Tensor, labels: flow.Tensor):
        return self.margin_loss(logits, labels) * self.scale

        
class ArcFaceFC(flow.nn.Module):
    def __init__(self, embedding_size, num_classes, scale, margin) -> None:
        super().__init__()
        self.fc = FC7(embedding_size, num_classes, None, False, False)
        self.head = ArcFace(scale, margin)
        self.loss = flow.nn.CrossEntropyLoss()

    def forward(self, features, labels):
        logits, labels = self.fc(features, labels)
        logits = self.head(logits, labels)
        loss = self.loss(logits, labels)
        return loss

if __name__ == "__main__":
    fc = ArcFaceFC(128, 100, 64, 0.5)
    features = flow.randn(4, 128).requires_grad_()
    labels = flow.randint(0, 100, (4, ))
    fc(features, labels).backward()

