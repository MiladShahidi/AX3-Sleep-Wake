project_config = {
    'subject_ids': [id for id in range(1, 36 + 1) if id != 27],
    'AX3_freq': 100,  # Hz
    'seconds_per_epoch': 30,
    'window_size': 1,
    'downsample_by': 5,  # Sample 1 out of every n measurements, e.g. 5 will down sample 100 hz to 20 hz

    'n_cv_folds': 12,

}

constants = {
    'SLEEP': 1,
    'WAKE': 0
}