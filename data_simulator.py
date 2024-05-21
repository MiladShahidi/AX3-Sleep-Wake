import pandas as pd
import tensorflow as tf
import numpy as np


def generate_fake_data(n_subjects, n_epochs, epoch_length):

    data = pd.DataFrame()
    n_rows = n_subjects * n_epochs

    labels = [np.round(np.random.uniform(0, 1)) for _ in range(n_rows)]  # label needs to be float
    
    X = [list(map(lambda x: round(x, 2), np.random.normal(scale=(label + 1) ** 1, size=epoch_length))) for label in labels]
    Y = [list(map(lambda x: round(x, 2), np.random.normal(scale=(label + 1) ** 1, size=epoch_length))) for label in labels]
    Z = [list(map(lambda x: round(x, 2), np.random.normal(scale=(label + 1) ** 1, size=epoch_length))) for label in labels]
    
    sub_id = [id for id in range(n_subjects) for epoch in range(n_epochs)]
    epoch_id = [epoch for id in range(n_subjects) for epoch in range(n_epochs)]

    data = pd.DataFrame({
        'subject_id': sub_id,
        'epoch_id': epoch_id,
        'X': X,
        'Y': Y,
        'Z': Z,
        'label': labels
    })

    # data = pd.concat([data, subject_data], ignore_index=True)
    
    return data


if __name__ == '__main__':

    df = generate_fake_data(
        n_subjects=3,
        n_epochs=10,
        epoch_length=20
    )

    print(df)