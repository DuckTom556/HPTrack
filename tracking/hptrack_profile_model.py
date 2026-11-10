import argparse
import importlib
import os

import torch
from torch import nn
from thop import profile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BackboneWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, templates, searches, annotations, history):
        features, _ = self.model.forward_backbone(
            templates, searches, annotations, history
        )
        return features


class HeadWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model.forward_head(back_out=features)["score_map"]


def estimate_chpu_macs(model, batch_size):
    backbone = model.backbone
    if getattr(backbone, "CHPU", None) is None:
        return 0

    seq_len = backbone.num_patches_search + 1
    total = 0
    for chpu in backbone.CHPU:
        mixer = chpu.layers[0].mixer
        d_model = mixer.d_model
        d_inner = mixer.d_inner
        d_state = mixer.d_state
        d_conv = mixer.d_conv
        dt_rank = mixer.dt_rank

        macs_per_token = (
                2 * d_model * d_inner
                + d_inner * (dt_rank + 2 * d_state)
                + dt_rank * d_inner
                + d_inner * d_model
                + d_inner * d_conv
                + 2 * d_inner * d_state
        )
        total += batch_size * seq_len * macs_per_token
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="hptrack")
    parser.add_argument("--config", default="hptrack_b256_got")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    yaml_path = os.path.join(
        ROOT, "experiments", args.script, f"{args.config}.yaml"
    )
    config_module = importlib.import_module(f"lib.config.{args.script}.config")
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_path)

    model_module = importlib.import_module(f"lib.models.{args.script}")
    model = model_module.build_hptrack(cfg).to(device).eval()

    template = torch.randn(
        1, 3, cfg.TEST.TEMPLATE_SIZE, cfg.TEST.TEMPLATE_SIZE, device=device
    )
    search = torch.randn(
        1, 3, cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_SIZE, device=device
    )
    n_templates = cfg.TEST.NUM_TEMPLATES
    templates = [template] * n_templates
    searches = [search]
    annotations = [
        torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device)
        for _ in range(n_templates)
    ]
    history = [None] * cfg.MODEL.BACKBONE.NUM_MFHP_LAYERS

    with torch.no_grad():
        macs_backbone, _ = profile(
            BackboneWrapper(model),
            inputs=(templates, searches, annotations, history),
            verbose=False,
        )
        features, _ = model.forward_backbone(
            templates, searches, annotations, history
        )
        macs_head, _ = profile(
            HeadWrapper(model), inputs=(features,), verbose=False
        )

    total_macs = macs_backbone + macs_head + estimate_chpu_macs(model, 1)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Params: {total_params / 1e6:.3f} M")
    print(f"MACs:   {total_macs / 1e9:.3f} G")


if __name__ == "__main__":
    main()
