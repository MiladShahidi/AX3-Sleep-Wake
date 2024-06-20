project_config = {
    'subject_ids': [id for id in range(1, 36 + 1) if id != 27],
    'AX3_freq': 100,  # Hz
    'seconds_per_epoch': 30,
    'window_size': 3,
    'n_cv_folds': 5
}

constants = {
    'SLEEP': 1,
    'WAKE': 0
}