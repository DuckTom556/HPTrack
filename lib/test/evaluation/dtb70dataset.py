
import os
import glob
import numpy as np

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class DTB70Dataset(BaseDataset):
    """DTB70 dataset consisting of 70 sequences."""

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.dtb70_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([
            self._construct_sequence(s) for s in self.sequence_list
        ])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name)

        # Load ground truth
        anno_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        ground_truth_rect = np.loadtxt(anno_path, delimiter=',', dtype=np.float64, ndmin=2)

        # Load frames
        frames_path = os.path.join(seq_path, 'img')
        frames_list = glob.glob(os.path.join(frames_path, '*.jpg'))

        # Sort frames numerically
        frames_list = sorted(
            frames_list,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

        assert len(frames_list) == len(ground_truth_rect), \
            f"{sequence_name}: frame/annotation count mismatch"

        return Sequence(
            sequence_name,
            frames_list,
            'dtb70',
            ground_truth_rect.reshape(-1, 4)
        )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = sorted([
            name for name in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, name))
            and os.path.isfile(
                os.path.join(self.base_path, name, 'groundtruth_rect.txt')
            )
        ])
        return sequence_list
