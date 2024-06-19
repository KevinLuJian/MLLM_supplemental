from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.TDIUC_dataset import TDIUCDataset, TDIUCEvalDataset


@registry.register_builder("TDIUC_dataset")
class TDIUCBuilder(BaseDatasetBuilder):
    train_dataset_cls = TDIUCDataset
    eval_dataset_cls = TDIUCEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/TDIUC/default_TDIUC.yaml"
    }
