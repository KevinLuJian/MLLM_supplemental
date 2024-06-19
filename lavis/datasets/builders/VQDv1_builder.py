from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.VQDv1_dataset import VQDv1Dataset, VQDv1EvalDataset


@registry.register_builder("vqdv1_dataset")
class VQDv1Builder(BaseDatasetBuilder):
    train_dataset_cls = VQDv1Dataset
    eval_dataset_cls = VQDv1EvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/VQDv1/defaults_VQDv1.yaml",
    }
