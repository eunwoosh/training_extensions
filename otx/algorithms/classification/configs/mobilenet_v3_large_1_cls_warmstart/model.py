"""MobileNet-V3-large-1 for multi-class config."""

# pylint: disable=invalid-name

_base_ = "../mobilenet_v3_large_1_cls_incr/model.py"

model = dict(
    type="BYOL",
    task="classification",
    base_momentum=0.996,
    neck=dict(
        type="SelfSLMLP",
        in_channels=960,
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=True
    ),
    head=dict(
        type='ConstrastiveHead',
        predictor=dict(
            type="SelfSLMLP",
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            with_avg_pool=False
        )
    )
)

custom_hooks = [
    dict(
        type='SelfSLHook',
        end_momentum=1.
    )
]

load_from = None

resume_from = None

fp16 = None
