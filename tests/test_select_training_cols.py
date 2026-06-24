def test_selected_columns_constant_exists():
    from config.const_training import SELECTED_COLUMNS
    assert isinstance(SELECTED_COLUMNS, list)
