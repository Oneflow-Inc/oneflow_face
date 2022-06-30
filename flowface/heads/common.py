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

class CombinedMarginLoss(flow.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.easy_margin = False


    def forward(self, logits, labels):
        index = flow.arange(len(labels), sbp=logits.sbp, placement=logits.placement).int()
        target_logits = logits[index, labels]
        sin_theta = flow.sqrt(1.0 - flow.pow(target_logits, 2))
        cos_theta_m = target_logits * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = flow.sign(target_logits - self.theta) / 2 + 0.5
        final_target_logits = cos_theta_m * mask + (1 - mask) * (target_logits * self.sinmm)
        logits =  flow.scatter(logits, 1, labels.reshape(-1, 1), final_target_logits.reshape(-1, 1))
        # logits[index, labels] = final_target_logits
        logits = logits * self.s

        return logits

class ArcFaceInsightface(flow.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFaceInsightface, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: flow.Tensor, labels: flow.Tensor):
        index = flow.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = flow.sqrt(1.0 - flow.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = flow.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = flow.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index]] = final_target_logit
        logits = logits * self.scale
        return logits

if __name__ == "__main__":
    fc = FC(128, 100)
    features = flow.randn(4, 128, requires_grad=True)
    labels = flow.randint(0, 100, (4,))
    logits, labels = fc(features, labels)
    logits.sum().backward()
