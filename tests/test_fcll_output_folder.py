"""Stage 1005 must not leave an empty output folder when it fails early.

Bug: `train_fcll_net` ran os.makedirs(1005-fcll_net) before loading the stage 506
sequences, so a run without them (or one that failed before training) left an empty
`data/output/1005-fcll_net/`. The folder is now created only once training succeeded.
"""
import os

from src.training_pipeline import TrainingPipeline
from config.const_training import FCLL_NET_OUTPUT_FOLDER


def test_missing_sequences_leave_no_output_folder(tmp_path):
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.project_root = str(tmp_path)

    result = pipeline.train_fcll_net()

    assert result["success"] is False
    assert not os.path.exists(os.path.join(str(tmp_path), FCLL_NET_OUTPUT_FOLDER))
