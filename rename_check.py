#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict

import torch


# 旧前缀 -> 新前缀，按顺序匹配，第一个命中就替换
RENAME_MAP = [
    # backbone.mamba_t.0/1/2...  ->  backbone.CHPU.0/1/2...
    ("backbone.mamba_t.",        "backbone.CHPU."),
    # backbone.mamba_s.ssm.      ->  backbone.HTFA.mamba.
    ("backbone.mamba_s.ssm.",    "backbone.HTFA.mamba."),
    # backbone.hp_token.0/1/2    ->  backbone.chp_tokens.0/1/2
    ("backbone.hp_token.",       "backbone.chp_tokens."),
]


def rename_keys(state_dict, rename_map):
    """按 rename_map 依次替换 key 前缀。"""
    new_state_dict = OrderedDict()
    changed = []

    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in rename_map:
            if new_k.startswith(old_prefix):
                new_k = new_prefix + new_k[len(old_prefix):]
                break  # 命中一个就停，避免连锁替换
        if new_k != k:
            changed.append((k, new_k))
        new_state_dict[new_k] = v

    return new_state_dict, changed


def find_state_dict(ckpt):
    """
    自动寻找 checkpoint 里的 state_dict。
    常见格式：
      - 直接就是 state_dict
      - {'model': state_dict, ...}
      - {'state_dict': state_dict, ...}
      - {'net': state_dict, ...}
      - {'network': state_dict, ...}
    """
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net", "network"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
    return ckpt, None


def main():
    parser = argparse.ArgumentParser(description="批量修改 checkpoint 中的参数名前缀")
    parser.add_argument(
        "src",
        nargs="?",
        default='/root/autodl-tmp/LAST/HPTrack/checkpoints/hptrack/hptrack_b256/ALL/HPTrack_ep0300.pth.tar',
        help="原始 .pth / .pth.tar 路径，不传则使用默认值"
    )
    parser.add_argument(
        "dst",
        nargs="?",
        default='/root/autodl-tmp/LAST/HPTrack/checkpoints/hptrack/hptrack_b256/HPTrack_ep0300.pth.tar',
        help="输出 .pth / .pth.tar 路径，不传则使用默认值"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要替换的键，不保存文件"
    )
    args = parser.parse_args()

    # PyTorch 2.6 默认 weights_only=True，这里显式关闭以兼容旧 checkpoint
    try:
        ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.src, map_location="cpu")

    state_dict, container_key = find_state_dict(ckpt)

    if not isinstance(state_dict, dict):
        raise TypeError("没有在 checkpoint 中找到 state_dict / dict 结构")

    print(f"从 {args.src} 加载，state_dict 容器 key = {container_key}")

    new_state_dict, changed = rename_keys(state_dict, RENAME_MAP)

    # 按前缀统计一下改了多少个
    from collections import Counter
    stat = Counter()
    for old_k, new_k in changed:
        for old_prefix, new_prefix in RENAME_MAP:
            if old_k.startswith(old_prefix):
                stat[f"{old_prefix}  ->  {new_prefix}"] += 1
                break

    print(f"\n共替换 {len(changed)} 个参数名：")
    for rule, cnt in stat.items():
        print(f"  [{cnt:3d}] {rule}")

    print("\n前 10 个替换示例：")
    for old_k, new_k in changed[:10]:
        print(f"  {old_k}")
        print(f"    -> {new_k}")
    if len(changed) > 10:
        print(f"  ... 还有 {len(changed) - 10} 个")

    if len(changed) == 0:
        print("\n警告：没有替换任何键，请检查 RENAME_MAP 是否正确。")
        print("当前 state_dict 前 20 个 key：")
        for k in list(state_dict.keys())[:20]:
            print("  ", k)

    # 写回容器
    if container_key is None:
        ckpt = new_state_dict
    else:
        ckpt[container_key] = new_state_dict

    if args.dry_run:
        print("\ndry-run 模式，未保存文件。")
    else:
        torch.save(ckpt, args.dst)
        print(f"\n已保存到: {args.dst}")


if __name__ == "__main__":
    main()