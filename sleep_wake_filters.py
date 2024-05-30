import pandas as pd
from config import constants


def filter_square_wave(square_wave):
    filtered_wave = square_wave.copy()
    consecutive_zeros = 0
    first_one_found = False
    for i, val in enumerate(square_wave):
        if val == 1:
            if not first_one_found:
                first_one_found = True
                consecutive_zeros = 0
                continue
            if consecutive_zeros > 0 and consecutive_zeros < 20:
                for j in range(i - consecutive_zeros, i):
                    filtered_wave[j] = 1
            consecutive_zeros = 0
        else:
            consecutive_zeros += 1
    if consecutive_zeros > 0 and consecutive_zeros < 20 and first_one_found:
        for j in range(len(square_wave) - consecutive_zeros, len(square_wave)):
            filtered_wave[j] = 1
    return filtered_wave


def filter_square_wave2 (square_wave):
    filtered_wave = square_wave.copy()
    consecutive_ones = 0
    for i, val in enumerate(square_wave):
        if val == 0:
            if consecutive_ones > 0 and consecutive_ones < 20:
                for j in range(i - consecutive_ones, i):
                    filtered_wave[j] = 0
            consecutive_ones = 0
        else:
            consecutive_ones += 1
    if consecutive_ones > 0 and consecutive_ones < 20:
        for j in range(len(square_wave) - consecutive_ones, len(square_wave)):
            filtered_wave[j] = 0
    return filtered_wave


def connect_bordering(square_wave):
        # Create a copy of the input array to modify
        modified_wave = square_wave.copy()
    
        # Iterate through the array, looking for transitions from 1 to 0
        for i in range(len(square_wave) - 1):
            if square_wave[i] == 1 and square_wave[i+1] == 0:
                # Found a transition - check if the gap is less than 120 values wide
                j = i + 1
                while j < len(square_wave) and square_wave[j] == 0:
                    j += 1
                gap_size = j - i - 1
                if gap_size < 120:
                    # Gap is small enough - connect the surrounding 1 values
                    for k in range(i+1, j):
                        modified_wave[k] = 1
    
        return modified_wave


# Vectorized filters

def episode_length(sleep_column):
    """
        Returns a Pandas Series with the same length as input
        each row of which shows the length of the current sleep/wake episode
    """
    local_df = pd.DataFrame({'sleep': sleep_column})
    
    # Creates a 0/1 change indicator, whether sleep/wake status changes in that time step
    local_df['change_ind'] = local_df['sleep'].diff().fillna(0).abs()  # fillna the first row
    # Creates a unique episode indicator (an increasing counter that numbers episodes)
    local_df['episode_ind'] = local_df['change_ind'].cumsum()
    # counts number of time steps in each episode to determine the length of that episode
    local_df['episode_len'] = local_df.groupby('episode_ind')['sleep'].transform('count')
    
    return local_df['episode_len'], local_df['episode_ind']


def generic_sleep_wake_filter(sleep_series, convert_to, min_len):
    convert_from = 1 - convert_to
    
    local_df = pd.DataFrame({'sleep': sleep_series})
    local_df['episode_len'], local_df['episode_ind'] = episode_length(local_df['sleep'])

    # Criteria for filtering
    filter = (local_df['sleep'] == convert_from) & (local_df['episode_len'] < min_len)
    
    # Pandas where(), unlinke np.where(), replaces values when condition is False, not True
    local_df['sleep'] = local_df['sleep'].where(~filter, convert_to)  # convert where filter is False
    
    return local_df['sleep']


def sleep_wake_filter_1(sleep_series):
    return generic_sleep_wake_filter(sleep_series, convert_to=constants['SLEEP'], min_len=10)


def sleep_wake_filter_2(sleep_series):
    return generic_sleep_wake_filter(sleep_series, convert_to=constants['WAKE'], min_len=10)


def sleep_wake_filter_3(sleep_series):
    return generic_sleep_wake_filter(sleep_series, convert_to=constants['SLEEP'], min_len=60)

