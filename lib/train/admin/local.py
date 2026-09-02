class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # 保存模型 checkpoint 的根目录
        self.tensorboard_dir = ''     # tensorboard 日志目录
        self.pretrained_networks =''# 预训练模型存放目录
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = ''
        self.coco_lmdb_dir = ''
        self.imagenet1k_dir = ''
        self.imagenet22k_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''