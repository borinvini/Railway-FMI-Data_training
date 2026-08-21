"""The day-grouped split needs a key to group on, and filter_columns drops
every identifier. This pins the one column that must survive."""
import pandas as pd
import pytest

from config.const_preprocessing import GROUP_KEY_COLUMN
from src.preprocessing_pipeline import PreprocessingPipeline


def test_filter_columns_keeps_the_group_key():
    df = pd.DataFrame({
        "trainDelayed": [0, 1],
        "Air temperature": [1.0, 2.0],
        GROUP_KEY_COLUMN: ["2018-01-01", "2018-01-02"],
        "operatorShortCode": ["vr", "vr"],
    })
    pipeline = PreprocessingPipeline()
    result = pipeline.filter_columns(df)
    assert GROUP_KEY_COLUMN in result.columns, (
        "the split cannot group by day without this column"
    )
    assert "operatorShortCode" not in result.columns, (
        "unrelated identifiers must still be dropped"
    )
