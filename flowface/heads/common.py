
import oneflow as flow
class FC(flow.nn.Module):
    def __init__(self, embedding_size, num_classes, sample_rate=1):
        super(FC, self).__init__()
        
        self.weight = flow.nn.Parameter(flow.empty(num_classes, embedding_size))
        print(self.weight.shape)
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)
        
        if sample_rate < 1:
            assert self.is_global, "Partial FC can only be used in global mode"
            self.sample = True
            size = flow.env.get_world_size()
            num_local = (num_classes + size - 1) // size
            self.num_sample = int(num_local * sample_rate)
            self.total_num_sample = self.num_sample * size
        else:
            self.sample = False
            


    def forward(self, x, label):
        x = flow.nn.functional.normalize(x, dim=1)
        if self.sample:
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
        x = flow.matmul(x, weight, transpose_b=True) + self.bias
        return x, label

if __name__ == "__main__":
    fc = FC(128, 100)
    features = flow.randn(4, 128, requires_grad=True)
    labels = flow.randint(0, 100, (4, ))
    logits, labels = fc(features, labels)
    logits.sum().backward()