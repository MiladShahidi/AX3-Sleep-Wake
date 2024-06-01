import pandas as pd
from config import constants


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


def filter_sleep_series(sleep_series):
    filters = [
        # Our epochs are 30-seconds long. So, 20 means 10 minutes and 120 means 60 minutes.
        lambda s: generic_sleep_wake_filter(s, convert_to=constants['WAKE'], min_len=20),
        lambda s: generic_sleep_wake_filter(s, convert_to=constants['SLEEP'], min_len=20),
        lambda s: generic_sleep_wake_filter(s, convert_to=constants['SLEEP'], min_len=120),
    ]

    for filter in filters:
        sleep_series = filter(sleep_series)

    return sleep_series
