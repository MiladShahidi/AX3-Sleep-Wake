project_config = {
    'subject_ids': [id for id in range(1, 36 + 1) if id != 27],
    'test_ids': [3, 11, 26, 31, 33],
    'AX3_freq': 100,  # Hz
    'seconds_per_epoch': 30,

    # Hyperparameters
    'down_sample_by': None,
    'num_conv_filters': 64,
    'num_attention_heads': 1,
    'stride': 2,
    'window_size': 21,
}

constants = {
    'SLEEP': 1,
    'WAKE': 0
}