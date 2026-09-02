from easydict import EasyDict as edict
import yaml
cfg = edict()
# MODE V1:EVP+VIM+CONF/V2:VIM+CONF/V3:LTHP+VIM+CONF
cfg.MODEL = edict()

#fastitpn
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "hivit_base" # encoder model
cfg.MODEL.BACKBONE.DROP_PATH = 0
cfg.MODEL.BACKBONE.PRETRAIN_FILE = "mae_hivit_base_1600ep.pth"
cfg.MODEL.BACKBONE.PRETRAINED_PATH = '../pretrained'
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.POS_TYPE = 'interpolate'
cfg.MODEL.BACKBONE.MULTIPLIER = 0.1
cfg.MODEL.BACKBONE.RETURN_INTER=False
cfg.MODEL.BACKBONE.CLS_TOKEN_LEN=1
cfg.MODEL.BACKBONE.FREEZE = False
cfg.MODEL.BACKBONE.GRAD_CKPT = True
cfg.MODEL.BACKBONE.TOKEN_TYPE_INDICATE = True#用于目标掩码
cfg.MODEL.BACKBONE.USE_CONF=False  #用于置信度头
cfg.MODEL.BACKBONE.USE_POOL=False
cfg.MODEL.BACKBONE.USE_HTFA=True #用于模板图像特征聚合
cfg.MODEL.BACKBONE.USE_MFHP=True#用于多级特征历史提示
cfg.MODEL.BACKBONE.USE_CHPU=True#用于多级特征历史提示
cfg.MODEL.BACKBONE.LAYER_INDEXES=[[10,18],[18,24],[24,30]]   # [[10,30]]#[[10,20],[20,30]]#[10,30]
cfg.MODEL.BACKBONE.NUM_MFHP_LAYERS=3
cfg.MODEL.BACKBONE.D_STATE = 16

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER" # MLP, CORNER, CENTER
cfg.MODEL.HEAD.NUM_CHANNELS = 256
cfg.MODEL.HEAD.CONF_LEN=1

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.CE_WEIGHT = 1.0 # weight for cross-entropy loss
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.CONF_WEIGHT=20.0
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.FIX_BN = False
cfg.TRAIN.BACKBONE_W = ""
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.TYPE = "normal" # normal, peft or fft
cfg.TRAIN.PRETRAINED_PATH = None

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATEF
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 2.0
cfg.TEST.SEARCH_SIZE = 128
cfg.TEST.EPOCH = 500
cfg.TEST.WINDOW = False # window penalty
cfg.TEST.NUM_TEMPLATES = 1
cfg.TEST.MEMORY_THRESHOLD=500

cfg.TEST.UPT = edict() #Update threshold
cfg.TEST.UPT.DEFAULT = 1
cfg.TEST.UPT.LASOT = 1
cfg.TEST.UPT.LASOT_EXTENSION_SUBSET = 1
cfg.TEST.UPT.TRACKINGNET = 1
cfg.TEST.UPT.TNL2K = 1
cfg.TEST.UPT.NFS = 1
cfg.TEST.UPT.UAV = 1
cfg.TEST.UPT.VOT20 = 1
cfg.TEST.UPT.GOT10K_TEST = 1
cfg.TEST.UPT.OTB=1

cfg.TEST.INTER = edict()#Update intervals
cfg.TEST.INTER.DEFAULT =999999
cfg.TEST.INTER.LASOT = 999999
cfg.TEST.INTER.LASOT_EXTENSION_SUBSET = 999999
cfg.TEST.INTER.TRACKINGNET = 999999
cfg.TEST.INTER.TNL2K = 999999
cfg.TEST.INTER.NFS = 999999
cfg.TEST.INTER.UAV = 999999
cfg.TEST.INTER.VOT20 = 999999
cfg.TEST.INTER.GOT10K_TEST = 999999
cfg.TEST.INTER.OTB = 999999

cfg.TEST.MB = edict() # memory_bank threshold 最大存储长度
cfg.TEST.MB.DEFAULT = 500
cfg.TEST.MB.LASOT = 500
cfg.TEST.MB.LASOT_EXTENSION_SUBSET = 500
cfg.TEST.MB.TRACKINGNET = 500
cfg.TEST.MB.TNL2K = 500
cfg.TEST.MB.NFS = 500
cfg.TEST.MB.UAV = 500
cfg.TEST.MB.VOT20 = 500
cfg.TEST.MB.GOT10K_TEST = 500
cfg.TEST.MB.OTB=500












def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


