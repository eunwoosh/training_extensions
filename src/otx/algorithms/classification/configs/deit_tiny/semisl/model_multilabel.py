"""deit-tiny config for semi-supervised multi-label classification."""

# pylint: disable=invalid-name

_base_ = ["../../../../../recipes/stages/classification/multilabel/semisl.yaml", "../../base/models/efficientnet.py"]
ckpt_url = "https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth"

model = dict(
    type="SemiSLMultilabelClassifier",
    task="classification",
    backbone=dict(arch="deit-tiny", init_cfg=dict(type="Pretrained", checkpoint=ckpt_url, prefix="backbone")),
    head=dict(
        type="SemiLinearMultilabelClsHead",
        use_dynamic_loss_weighting=True,
        unlabeled_coef=0.1,
        in_channels=-1,
        aux_mlp=dict(hid_channels=0, out_channels=1024),
        normalized=True,
        scale=7.0,
        loss=dict(type="AsymmetricAngularLossWithIgnore", gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        aux_loss=dict(
            type="BarlowTwinsLoss",
            off_diag_penality=1.0 / 128.0,
            loss_weight=1.0,
        ),
    ),
)

fp16 = None
