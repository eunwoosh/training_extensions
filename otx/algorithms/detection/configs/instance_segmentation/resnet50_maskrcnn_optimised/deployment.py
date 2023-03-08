"""MMDployment config of Resnet model for Instance-Seg Task."""

_base_ = ["../../base/deployments/base_instance_segmentation_dynamic.py"]

ir_config = dict(
    output_names=["boxes", "labels", "masks", "tile_prob"]
)

backend_config = dict(
    # dynamic batch causes forever running openvino process
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 800, 1344]))],
)

partition_config = dict(
    type='tile_classifier',
    apply_marks=True,
    partition_cfg=[
        dict(
            save_file='classifier.onnx',
            start=['tile_classifier:input'],
            end=['tile_classifier:output'],
            output_names=["prob"])])
