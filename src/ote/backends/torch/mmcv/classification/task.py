import os

import numpy as np
from mmcls.models.builder import build_from_cfg
from mmcv import ConfigDict

from ote.backends.torch.mmcv.config import update_or_add_custom_hook
from ote.backends.torch.mmcv.task import MMTask
from ote.core.config import Config
from ote.logger import get_logger

logger = get_logger()

CLASS_INC_DATASET = ['MPAClsDataset', 'MPAMultilabelClsDataset', 'MPAHierarchicalClsDataset',
                     'ClsDirDataset', 'ClsTVDataset']
PSEUDO_LABEL_ENABLE_DATASET = ['ClassIncDataset', 'LwfTaskIncDataset', 'ClsTVDataset']
WEIGHT_MIX_CLASSIFIER = ['SAMImageClassifier']

class MMClsTask(MMTask):
    def configure(self, cfg: Config, model_cfg=None, data_cfg=None, **kwargs):
        logger.info(f"configure()")
        training = kwargs.get("training", True)

        if model_cfg is not None:
            if hasattr(cfg, "model"):
                cfg.merge_from_dict(dict(model=model_cfg._cfg_dict))
            else:
                cfg.model = copy.deepcopy(model_cfg._cfg_dict)

        if cfg.model.pop('task', None) != 'classification':
            raise ValueError(
                f'Given model_cfg ({cfg.model.type}) is not supported by classification recipe'
            )

        model_ckpt = kwargs.get("model_ckpt")
        if model_ckpt is not None:
            cfg.load_from = MMTask.get_model_ckpt(model_ckpt)

        self.configure_model(cfg, training, **kwargs)

        # OMZ-plugin
        if cfg.model.backbone.type == 'OmzBackboneCls':
            ir_path = kwargs.get('ir_path', None)
            if ir_path is None:
                raise RuntimeError('OMZ model needs OpenVINO bin/XML files.')
            cfg.model.backbone.model_path = ir_path

        pretrained = kwargs.get('pretrained', None)
        if pretrained and isinstance(pretrained, str):
            logger.info(f'Overriding cfg.load_from -> {pretrained}')
            cfg.load_from = pretrained

        # Data
        if data_cfg is not None:
            if hasattr(cfg, "data"):
                cfg.merge_from_dict(dict(data=data_cfg._cfg_dict))
            else:
                cfg.data = copy.deepcopy(data_cfg.data)

        self.configure_data(cfg, training, **kwargs)

         # Task
        if 'task_adapt' in cfg:
            model_meta = self.get_model_meta(cfg)
            model_tasks, dst_classes = self.configure_task(cfg, training, model_meta, **kwargs)
            if model_tasks is not None:
                self.model_tasks = model_tasks
            if dst_classes is not None:
                self.model_classes = dst_classes
        else:
            if 'num_classes' not in cfg.data:
                cfg.data.num_classes = len(cfg.data.train.get('classes', []))
            cfg.model.head.num_classes = cfg.data.num_classes

        if cfg.model.head.get('topk', False) and isinstance(cfg.model.head.topk, tuple):
            cfg.model.head.topk = (1,) if cfg.model.head.num_classes < 5 else (1, 5)
            if cfg.model.get('multilabel', False) or cfg.model.get('hierarchical', False):
                cfg.model.head.pop('topk', None)

        # Other hyper-parameters
        if cfg.get('hyperparams', False):
            self.configure_hyperparams(cfg, training, **kwargs)
        logger.info(f"updated config = {cfg}")
        return cfg

    @staticmethod
    def configure_task(cfg, training, model_meta=None, **kwargs):
        """Configure for Task Adaptation Task
        """
        logger.info("configuire_task()")
        task_adapt_type = cfg['task_adapt'].get('type', None)
        adapt_type = cfg['task_adapt'].get('op', 'REPLACE')

        model_tasks, dst_classes = None, None
        model_classes, data_classes = [], []
        train_data_cfg = MMTask.get_train_data_cfg(cfg)
        if isinstance(train_data_cfg, list):
            train_data_cfg = train_data_cfg[0]

        model_classes = MMClsTask.get_model_classes(cfg)
        data_classes = MMClsTask.get_data_classes(cfg)
        if model_classes:
            cfg.model.head.num_classes = len(model_classes)
        elif data_classes:
            cfg.model.head.num_classes = len(data_classes)
        model_meta['CLASSES'] = model_classes

        if not train_data_cfg.get('new_classes', False):  # when train_data_cfg doesn't have 'new_classes' key
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg['new_classes'] = new_classes

        if training:
            # if Trainer to Stage configure, training = True
            if train_data_cfg.get('tasks'):
                # Task Adaptation
                if model_meta.get('tasks', False):
                    model_tasks, old_tasks = refine_tasks(train_data_cfg, model_meta, adapt_type)
                else:
                    raise KeyError(f'can not find task meta data from {cfg.load_from}.')
                cfg.model.head.update({'old_tasks': old_tasks})
                # update model.head.tasks with training dataset's tasks if it's configured as None
                if cfg.model.head.get('tasks') is None:
                    logger.info("'tasks' in model.head is None. updated with configuration on train data "
                                f"{train_data_cfg.get('tasks')}")
                    cfg.model.head.update({'tasks': train_data_cfg.get('tasks')})
            elif 'new_classes' in train_data_cfg:
                # Class-Incremental
                dst_classes, old_classes = refine_cls(train_data_cfg, data_classes, model_meta, adapt_type)
            else:
                raise KeyError(
                    '"new_classes" or "tasks" should be defined for incremental learning w/ current model.'
                )

            if task_adapt_type == 'mpa':
                if train_data_cfg.type not in CLASS_INC_DATASET:  # task incremental is not supported yet
                    raise NotImplementedError(
                        f'Class Incremental Learning for {train_data_cfg.type} is not yet supported!')

                if cfg.model.type in WEIGHT_MIX_CLASSIFIER:
                    cfg.model.task_adapt = ConfigDict(
                        src_classes=model_classes,
                        dst_classes=data_classes,
                    )

                # Train dataset config update
                train_data_cfg.classes = dst_classes

                # model configuration update
                cfg.model.head.num_classes = len(dst_classes)

                if not cfg.model.get('multilabel', False) and not cfg.model.get('hierarchical', False):
                    efficient_mode = cfg['task_adapt'].get('efficient_mode', True)
                    gamma = 2 if efficient_mode else 3
                    sampler_type = 'balanced'

                    if len(set(model_classes) & set(dst_classes)) == 0 or set(model_classes) == set(dst_classes):
                        cfg.model.head.loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
                    else:
                        cfg.model.head.loss = ConfigDict(
                            type='SoftmaxFocalLoss',
                            loss_weight=1.0,
                            gamma=gamma,
                            reduction='none',
                        )
                else:
                    efficient_mode = cfg['task_adapt'].get('efficient_mode', False)
                    sampler_type = 'cls_incr'

                if len(set(model_classes) & set(dst_classes)) == 0 or set(model_classes) == set(dst_classes):
                    sampler_flag = False
                else:
                    sampler_flag = True

                # Update Task Adapt Hook
                task_adapt_hook = ConfigDict(
                    type='TaskAdaptHook',
                    src_classes=old_classes,
                    dst_classes=dst_classes,
                    model_type=cfg.model.type,
                    sampler_flag=sampler_flag,
                    sampler_type=sampler_type,
                    efficient_mode=efficient_mode
                )
                update_or_add_custom_hook(cfg, task_adapt_hook)

        else:  # if not training phase (eval)
            if train_data_cfg.get('tasks'):
                if model_meta.get('tasks', False):
                    cfg.model.head['tasks'] = model_meta['tasks']
                else:
                    raise KeyError(f'can not find task meta data from {cfg.load_from}.')
            elif train_data_cfg.get('new_classes'):
                dst_classes, _ = refine_cls(train_data_cfg, data_classes, model_meta, adapt_type)
                cfg.model.head.num_classes = len(dst_classes)

        # Pseudo label augmentation
        pre_stage_res = kwargs.get('pre_stage_res', None)
        if pre_stage_res:
            logger.info(f'pre-stage dataset: {pre_stage_res}')
            if train_data_cfg.type not in PSEUDO_LABEL_ENABLE_DATASET:
                raise NotImplementedError(
                    f'Pseudo label loading for {train_data_cfg.type} is not yet supported!')
            train_data_cfg.pre_stage_res = pre_stage_res
            if train_data_cfg.get('tasks'):
                train_data_cfg.model_tasks = model_tasks
                cfg.model.head.old_tasks = old_tasks
            elif train_data_cfg.get('CLASSES'):
                train_data_cfg.dst_classes = dst_classes
                cfg.data.val.dst_classes = dst_classes
                cfg.data.test.dst_classes = dst_classes
                cfg.model.head.num_classes = len(train_data_cfg.dst_classes)
                cfg.model.head.num_old_classes = len(old_classes)
        return model_tasks, dst_classes

    @staticmethod
    def configure_model(cfg, training, **kwargs):
        # verify and update model configurations
        # check whether in/out of the model layers require updating
        logger.info("configuire_model()")
        if cfg.get('load_from', None) and cfg.model.backbone.get('pretrained', None):
            cfg.model.backbone.pretrained = None

        update_required = False
        if cfg.model.get('neck') is not None:
            if cfg.model.neck.get('in_channels') is not None and cfg.model.neck.in_channels <= 0:
                update_required = True
        if not update_required and cfg.model.get('head') is not None:
            if cfg.model.head.get('in_channels') is not None and cfg.model.head.in_channels <= 0:
                update_required = True
        if not update_required:
            return

        # update model layer's in/out configuration
        input_shape = [3, 224, 224]
        logger.debug(f'input shape for backbone {input_shape}')
        from mmcls.models.builder import BACKBONES as backbone_reg
        layer = build_from_cfg(cfg.model.backbone, backbone_reg)
        output = layer(torch.rand([1] + input_shape))
        if isinstance(output, (tuple, list)):
            output = output[-1]
        output = output.shape[1]
        if cfg.model.get('neck') is not None:
            if cfg.model.neck.get('in_channels') is not None:
                logger.info(f"'in_channels' config in model.neck is updated from "
                            f"{cfg.model.neck.in_channels} to {output}")
                cfg.model.neck.in_channels = output
                input_shape = [i for i in range(output)]
                logger.debug(f'input shape for neck {input_shape}')
                from mmcls.models.builder import NECKS as neck_reg
                layer = build_from_cfg(cfg.model.neck, neck_reg)
                output = layer(torch.rand([1] + input_shape))
                if isinstance(output, (tuple, list)):
                    output = output[-1]
                output = output.shape[1]
        if cfg.model.get('head') is not None:
            if cfg.model.head.get('in_channels') is not None:
                logger.info(f"'in_channels' config in model.head is updated from "
                            f"{cfg.model.head.in_channels} to {output}")
                cfg.model.head.in_channels = output

            # checking task incremental model configurations

    @staticmethod
    def configure_data(cfg, training, **kwargs):
        logger.info('configure_data()')
        # update data configuration using image options
        def configure_split(target):

            def update_transform(opt, pipeline, idx, transform):
                if isinstance(opt, dict):
                    if '_delete_' in opt.keys() and opt.get('_delete_', False):
                        # if option include _delete_=True, remove this transform from pipeline
                        logger.info(f"configure_data: {transform['type']} is deleted")
                        del pipeline[idx]
                        return
                    logger.info(f"configure_data: {transform['type']} is updated with {opt}")
                    transform.update(**opt)

            def update_config(src, pipeline_options):
                logger.info(f'update_config() {pipeline_options}')
                if src.get('pipeline') is not None or \
                        (src.get('dataset') is not None and src.get('dataset').get('pipeline') is not None):
                    if src.get('pipeline') is not None:
                        pipeline = src.get('pipeline', None)
                    else:
                        pipeline = src.get('dataset').get('pipeline')
                    if isinstance(pipeline, list):
                        for idx, transform in enumerate(pipeline):
                            for opt_key, opt in pipeline_options.items():
                                if transform['type'] == opt_key:
                                    update_transform(opt, pipeline, idx, transform)
                    elif isinstance(pipeline, dict):
                        for _, pipe in pipeline.items():
                            for idx, transform in enumerate(pipe):
                                for opt_key, opt in pipeline_options.items():
                                    if transform['type'] == opt_key:
                                        update_transform(opt, pipe, idx, transform)
                    else:
                        raise NotImplementedError(f'pipeline type of {type(pipeline)} is not supported')
                else:
                    logger.info('no pipeline in the data split')

            split = cfg.data.get(target)
            if split is not None:
                if isinstance(split, list):
                    for sub_item in split:
                        update_config(sub_item, pipeline_options)
                elif isinstance(split, dict):
                    update_config(split, pipeline_options)
                else:
                    logger.warning(f"type of split '{target}'' should be list or dict but {type(split)}")

        logger.debug(f'[args] {cfg.data}')
        pipeline_options = cfg.data.pop('pipeline_options', None)
        if pipeline_options is not None and isinstance(pipeline_options, dict):
            configure_split('train')
            configure_split('val')
            if not training:
                configure_split('test')
            configure_split('unlabeled')

    @staticmethod
    def configure_hyperparams(cfg, training, **kwargs):
        logger.info('configure_hyperparams()')
        hyperparams = kwargs.get('hyperparams', None)
        if hyperparams is not None:
            bs = hyperparams.get('bs', None)
            if bs is not None:
                cfg.data.samples_per_gpu = bs

            lr = hyperparams.get('lr', None)
            if lr is not None:
                cfg.optimizer.lr = lr

    @staticmethod
    def get_data_classes(cfg):
        data_classes = []
        train_cfg = MMTask.get_train_data_cfg(cfg)
        if 'data_classes' in train_cfg:
            data_classes = list(train_cfg.pop('data_classes', []))
        elif 'classes' in train_cfg:
            data_classes = list(train_cfg.classes)
        return data_classes

    @staticmethod
    def get_model_classes(cfg):
        """Extract trained classes info from checkpoint file.
        MMCV-based models would save class info in ckpt['meta']['CLASSES']
        For other cases, try to get the info from cfg.model.classes (with pop())
        - Which means that model classes should be specified in model-cfg for
          non-MMCV models (e.g. OMZ models)
        """
        classes = []
        meta = MMTask.get_model_meta(cfg)
        # for MPA classification legacy compatibility
        classes = meta.get('CLASSES', [])
        classes = meta.get('classes', classes)

        if len(classes) == 0:
            ckpt_path = cfg.get('load_from', None)
            if ckpt_path:
                classes = MMTask.read_label_schema(ckpt_path)
        if len(classes) == 0:
            classes = cfg.model.pop('classes', cfg.pop('model_classes', []))
        return classes


def refine_tasks(train_cfg, meta, adapt_type):
    new_tasks = train_cfg['tasks']
    if adapt_type == 'REPLACE':
        old_tasks = {}
        model_tasks = new_tasks
    elif adapt_type == 'MERGE':
        old_tasks = meta['tasks']
        model_tasks = copy.deepcopy(old_tasks)
        for task, cls in new_tasks.items():
            if model_tasks.get(task):
                model_tasks[task] = model_tasks[task] \
                                            + [c for c in cls if c not in model_tasks[task]]
            else:
                model_tasks.update({task: cls})
    else:
        raise KeyError(f'{adapt_type} is not supported for task_adapt options!')
    return model_tasks, old_tasks


def refine_cls(train_cfg, data_classes, meta, adapt_type):
    # Get 'new_classes' in data.train_cfg & get 'old_classes' pretreained model meta data CLASSES
    new_classes = train_cfg['new_classes']
    old_classes = meta['CLASSES']
    if adapt_type == 'REPLACE':
        # if 'REPLACE' operation, then dst_classes -> data_classes
        dst_classes = data_classes.copy()
    elif adapt_type == 'MERGE':
        # if 'MERGE' operation, then dst_classes -> old_classes + new_classes (merge)
        dst_classes = old_classes + [cls for cls in new_classes if cls not in old_classes]
    else:
        raise KeyError(f'{adapt_type} is not supported for task_adapt options!')
    return dst_classes, old_classes