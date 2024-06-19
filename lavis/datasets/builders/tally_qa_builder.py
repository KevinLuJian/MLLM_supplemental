from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.tallyqa_datasets import TallyQADataset,TallyQAEvalDataset


@registry.register_builder("tallyqa_dataset")
class TallyqaBuilder(BaseDatasetBuilder):
    train_dataset_cls = TallyQADataset
    eval_dataset_cls = TallyQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/tallyqa/default_tallyqa.yaml",
    }
