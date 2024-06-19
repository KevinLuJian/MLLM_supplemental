from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.DVQA_dataset import DVQADataset


@registry.register_builder("dvqa_dataset")
class DVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = DVQADataset
    eval_dataset_cls = DVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/DVQA/defaults_DVQA.yaml",
    }


