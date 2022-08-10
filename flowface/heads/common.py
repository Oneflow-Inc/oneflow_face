import math

import oneflow as flow


class FC(flow.nn.Module):
    def __init__(
        self, embedding_size, num_classes, is_global=True, is_parallel=True, sample_rate=1
    ):
        super(FC, self).__init__()
        # placement = flow.env.all_device_placement("cuda")
        if is_global:
            placement = flow.env.all_device_placement("cuda")
            sbp = flow.sbp.split(0) if is_parallel else flow.sbp.broadcast
        else:
            placement = None
            sbp = None
        self.weight = flow.nn.Parameter(
            flow.empty(num_classes, embedding_size, sbp=sbp, placement=placement)
        )
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)

        if sample_rate < 1:
            # TODO: support broadcast
            assert is_parallel, "Partial FC doesn't support broadcast yet"
            assert is_global, "Partial FC can only be used in global mode"
            num_sample = math.ceil(num_classes * sample_rate)
            self.sampler = flow.nn.DistributedPariticalFCSample(num_sample)
        else:
            self.sampler = False

    def forward(self, x, label):
        x = flow.nn.functional.normalize(x, dim=1)
        if self.sampler:
            (
                mapped_label,
                sampled_label,
                sampled_weight,
            ) = self.sampler(self.weight, label)
            label = mapped_label
            weight = sampled_weight
        else:
            weight = self.weight
        weight = flow.nn.functional.normalize(weight, dim=1)
        x = flow.matmul(x, weight, transpose_b=True)
        return x, label


if __name__ == "__main__":
    fc = FC(128, 100)
    features = flow.randn(4, 128, requires_grad=True)
    labels = flow.randint(0, 100, (4,))
    logits, labels = fc(features, labels)
    logits.sum().backward()
