import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

import math

class SphereFace2(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition>
        margin='C' -> SphereFace2-C
        margin='A' -> SphereFace2-A
        marign='M' -> SphereFAce2-M
    """
    def __init__(self, num_classes, embedding_size, is_global=True, is_parallel=False, magn_type='C',
            alpha=0.7, r=40., m=0.4, t=3., lw=50.):
        super().__init__()
        self.feat_dim = embedding_size
        self.num_class = num_classes
        self.magn_type = magn_type

        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        placement = flow.env.all_device_placement("cuda")
        if is_global:
            sbp_w = flow.sbp.split(0) if is_parallel else flow.sbp.broadcast
            sbp_b = flow.sbp.broadcast
        else:
            sbp_w = sbp_b = None
        self.w = flow.nn.Parameter(flow.empty(num_classes, embedding_size, sbp=sbp_w, placement=placement))
        self.b = nn.Parameter(flow.empty((1, 1), sbp=sbp_b, placement=placement))

        # init weights
        if is_global:
            placement = flow.env.all_device_placement("cuda")
            self.w = nn.Parameter(flow.empty(num_classes, embedding_size, sbp=flow.sbp.split(0), placement=placement))
            self.b = nn.Parameter(flow.empty((1, 1), sbp=flow.sbp.broadcast, placement=placement))
        else:
            self.w = nn.Parameter(flow.empty(num_classes, embedding_size))
            self.b = nn.Parameter(flow.Tensor(1, 1))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1. - alpha) * (num_classes - 1.))
        if magn_type == 'C':
            ay = r * (2. * 0.5**t - 1. - m)
            ai = r * (2. * 0.5**t - 1. + m)
        elif magn_type == 'A':
            theta_y = min(math.pi, math.pi/2. + m)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)
            ai = r * (2. * 0.5**t - 1.)
        elif magn_type == 'M':
            theta_y = min(math.pi, m * math.pi/2.)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)
            ai = r * (2. * 0.5**t - 1.)
        else:
            raise NotImplementedError

        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
        b = (math.log(2. * z) - ai
             - math.log(1. - z +  math.sqrt(temp)))
            # placement = flow.env.all_device_placement("cuda")
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        # import ipdb; ipdb.set_trace()
        with flow.no_grad():
            self.w.data = F.normalize(self.w.data, dim=1)

        cos_theta = flow.matmul(F.normalize(x, dim=1), self.w.T)
        #delta theta with margin

        one_hot = flow.zeros_like(cos_theta)
        one_hot = flow.scatter(one_hot, 1, y.view(-1, 1), 1.)
        # one_hot.scatter_(1, y.view(-1, 1), 1.)
        with flow.no_grad():
            if self.magn_type == 'C':
                g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.
                g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)
            elif self.magn_type == 'A':
                theta_m = flow.acos(cos_theta.clamp(-1+1e-5, 1.-1e-5))
                # theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
                theta_m = flow.scatter(theta_m, 1, y.view(-1, 1), self.m, reduce='add')
                theta_m.clamp_(1e-5, 3.14159)
                g_cos_theta = flow.cos(theta_m)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            elif self.magn_type == 'M':
                m_theta = flow.acos(cos_theta.clamp(-1+1e-5, 1.-1e-5))
                # m_theta.scatter_(1, y.view(-1, 1), self.m, reduce='multiply')
                m_theta = flow.scatter(m_theta, 1, y.view(-1, 1), self.m, reduce='multiply')
                m_theta.clamp_(1e-5, 3.14159)
                g_cos_theta = flow.cos(m_theta)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta
        
        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        loss = flow._C.binary_cross_entropy_with_logits_loss(
            logits, one_hot, weight, None, "mean")

        return loss


if __name__ == "__main__":
    fc = SphereFace2(128, 100)
    features = flow.randn(4, 128).requires_grad_()
    labels = flow.randint(0, 100, (4, ))
    fc(features, labels).backward()