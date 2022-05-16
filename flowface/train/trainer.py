import logging
import math

import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp

from flowface.backbones import get_model
from flowface.heads import get_head
from flowface.utils.losses import CrossEntropyLoss_sbp
from flowface.utils.ofrecord_data_utils import OFRecordDataLoader, SyntheticDataLoader
from flowface.utils.utils_callbacks import (
    CallBackLogging,
    CallBackModelCheckpoint,
    CallBackVerification,
)
from flowface.utils.utils_logging import AverageMeter

from .graph_def import EvalGraph, TrainGraph


def make_data_loader(args, mode, is_global=False, synthetic=False):
    assert mode in ("train", "validation")

    if mode == "train":
        total_batch_size = args.batch_size * flow.env.get_world_size()
        batch_size = args.batch_size
        num_samples = args.num_image
    else:
        total_batch_size = args.val_global_batch_size
        batch_size = args.val_batch_size
        num_samples = args.val_samples_per_epoch

    placement = None
    sbp = None

    if is_global:
        placement = flow.env.all_device_placement("cpu")
        sbp = flow.sbp.split(0)
        batch_size = total_batch_size

    if synthetic:

        data_loader = SyntheticDataLoader(
            batch_size=batch_size,
            num_classes=args.num_classes,
            placement=placement,
            sbp=sbp,
            channel_last=args.channel_last,
        )
        return data_loader.to("cuda")

    ofrecord_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode=mode,
        dataset_size=num_samples,
        batch_size=batch_size,
        total_batch_size=total_batch_size,
        data_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
        channel_last=args.channel_last,
        use_gpu_decode=args.use_gpu_decode,
    )
    return ofrecord_data_loader


def make_optimizer(args, model):
    param_group = {"params": [p for p in model.parameters() if p is not None]}

    optimizer = flow.optim.SGD(
        [param_group],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer

class Train_Module(flow.nn.Module):
    def __init__(self, backbone, head, num_classes, embedding_size, model_parallel, is_global, sample_rate):
        super(Train_Module, self).__init__()
        if is_global:
            self.backbone = backbone.to_global(sbp=flow.sbp.broadcast, placement=flow.env.all_device_placement("cuda"))
        else:
            self.backbone = backbone
        self.head = head(num_classes, embedding_size, is_global=is_global, is_parallel=model_parallel, sample_rate=sample_rate)

    def forward(self, imgs, labels):
        feature = self.backbone(imgs)
        loss = self.head(feature, labels)
        return loss

class Trainer(object):
    def __init__(self, cfg, load_path, world_size, rank):
        """
        Args:
            cfg (easydict.EasyDict): train configs.
            placement (_oneflow_internal.placement):train devices
            load_path (str) : pretrained model path
            world_size (int): total number of all devices
            rank (int)      : local device number
        """

        self.load_path = load_path
        self.cfg = cfg
        self.world_size = world_size
        self.rank = rank

        # model
        self.backbone = get_model(cfg.network)(**cfg.network_kwargs).to("cuda")
        self.head = get_head(cfg.head)
        self.train_module = Train_Module(self.backbone, self.head, cfg.num_classes, cfg.embedding_size, cfg.model_parallel, cfg.is_global, cfg.sample_rate).to("cuda")
        if cfg.resume:
            if load_path is not None:
                self.load_state_dict()
            else:
                logging.info("Model resume failed! load path is None ")

        # optimizer
        self.optimizer = make_optimizer(cfg, self.train_module)

        # data
        self.train_data_loader = make_data_loader(
            cfg, "train", self.cfg.is_global, self.cfg.synthetic
        )

        # loss

        self.of_cross_entropy = CrossEntropyLoss_sbp()

        # lr_scheduler
        self.decay_step = self.cal_decay_step()
        self.scheduler = flow.optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer, milestones=self.decay_step, gamma=0.1
        )

        # log
        self.callback_logging = CallBackLogging(
            cfg.log_frequent, rank, cfg.total_step, cfg.batch_size, world_size, None
        )
        # val
        self.callback_verification = CallBackVerification(
            cfg.val_frequence,
            rank,
            cfg.val_targets,
            cfg.ofrecord_path,
            is_global=self.cfg.is_global,
        )
        # save checkpoint
        self.callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

        self.losses = AverageMeter()
        self.start_epoch = 0
        self.global_step = 0

    def __call__(self):
        # Train
        if self.cfg.graph:
            self.train_graph()
        else:
            if not self.cfg.is_global:
                self.train_module = ddp(self.train_module)
            self.train_eager()

    def load_state_dict(self):
        if self.is_global:
            state_dict = flow.load(self.load_path, global_src_rank=0)
        elif self.rank == 0:
            state_dict = flow.load(self.load_path)
        else:
            return
        logging.info("Model resume successfully!")
        self.model.load_state_dict(state_dict)

    def cal_decay_step(self):
        cfg = self.cfg
        num_image = cfg.num_image
        total_batch_size = cfg.batch_size * self.world_size
        self.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
        self.cfg.total_step = num_image // total_batch_size * cfg.num_epoch
        logging.info("Total Step is:%d" % self.cfg.total_step)
        return [x * num_image // total_batch_size for x in cfg.decay_epoch]

    def train_graph(self):
        train_graph = TrainGraph(
            self.train_module,
            self.cfg,
            self.train_data_loader,
            self.optimizer,
            self.scheduler,
        )
        # train_graph.debug()
        val_graph = EvalGraph(self.backbone, self.cfg)

        for epoch in range(self.start_epoch, self.cfg.num_epoch):
            self.train_module.train()
            one_epoch_steps = len(self.train_data_loader)
            for steps in range(one_epoch_steps):
                self.global_step += 1
                loss = train_graph()
                loss = loss.to_global(sbp=flow.sbp.broadcast).to_local().numpy()
                self.losses.update(loss, 1)
                self.callback_logging(
                    self.global_step,
                    self.losses,
                    epoch,
                    False,
                    self.scheduler.get_last_lr()[0],
                )
                self.callback_verification(self.global_step, self.backbone, val_graph)
                if self.global_step >= self.cfg.train_num:
                    exit(0)
            self.callback_checkpoint(self.global_step, epoch, self.backbone, is_global=True)

    def train_eager(self):
        for epoch in range(self.start_epoch, self.cfg.num_epoch):
            self.train_module.train()

            one_epoch_steps = len(self.train_data_loader)
            for steps in range(one_epoch_steps):
                self.global_step += 1
                image, label = self.train_data_loader()
                image = image.to("cuda")
                label = label.to("cuda")

                loss = self.train_module(image, label)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss = loss.numpy()
                self.losses.update(loss, 1)
                self.callback_logging(
                    self.global_step,
                    self.losses,
                    epoch,
                    False,
                    self.scheduler.get_last_lr()[0],
                )
                self.callback_verification(self.global_step, self.backbone)
                self.scheduler.step()
                if self.global_step >= self.cfg.train_num:
                    exit(0)
            self.callback_checkpoint(
                self.global_step, epoch, self.train_module, is_global=self.cfg.is_global
            )
