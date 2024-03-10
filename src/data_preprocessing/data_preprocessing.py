import pandas as pd
import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv
import tensorflow as tf


COLUMNS = ['episode', 'step', 'pos', 'vel', 'action']
TRAIN_PERCENTAGE = 0.7
DEV_PERCENTAGE = 0.1
EPISODES = 200
STEPS = 200


def get_normalizations(columns, data_dir):
    """
    Get mean and std of certain columns of a given .csv file.
    :param columns: Required columns.
    :param data_dir: Path of data.
    :return: Mean and std.
    """
    df = pd.read_csv(data_dir)
    selected = df[columns]
    mean, std = selected.mean(), selected.std()
    return np.array(mean), np.array(std)


class MountainCarNew(MountainCarEnv):
    """
    Define two additional reset functions.
    """
    def __init__(self):
        super(MountainCarNew, self).__init__()

    def reset_all_pos(self, low=-1.2, high=0.6):
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])
        return np.array(self.state)

    def reset_nn(self, window_size, state_buffer, min_pos=-0.6, max_pos=-0.4):
        pos, vel = self.reset_all_pos(low=min_pos, high=max_pos)
        action = self.action_space.sample()
        state_buffer.append(np.float64([pos, vel, action]))
        for _ in range(window_size-1):
            obs, _, _, _ = self.step(action)
            action = self.action_space.sample()
            state_buffer.append(np.float64([obs[0], obs[1], action]))


def data_sampling(out_dir, episodes=EPISODES):
    """
    Sample dataset.
    :param out_dir: Output directory.
    :param episodes: Running episodes.
    """
    env = MountainCarNew()
    env.np_random.seed(0)
    samples = []

    for episode in range(episodes):
        obs = env.reset_all_pos()
        # obs = env.reset()
        step = 0
        done = False
        while step < STEPS and not done:
            action = env.action_space.sample()
            samples.append({COLUMNS[0]: episode,
                            COLUMNS[1]: step,
                            COLUMNS[2]: obs[0],
                            COLUMNS[3]: obs[1],
                            COLUMNS[4]: action})
            step += 1
            obs, reward, done, _ = env.step(action)
    env.close()

    df = pd.DataFrame(samples, columns=COLUMNS)

    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()

    grouped = df.groupby(([COLUMNS[0]]))
    df_l = [x[1] for x in grouped]
    n = len(grouped)

    for i in df_l[0:int(n * TRAIN_PERCENTAGE)]:
        train_df = pd.concat([train_df, i], ignore_index=True)

    for i in df_l[int(n * TRAIN_PERCENTAGE):int(n * (TRAIN_PERCENTAGE+DEV_PERCENTAGE))]:
        dev_df = pd.concat([dev_df, i], ignore_index=True)

    for i in df_l[int(n * (TRAIN_PERCENTAGE+DEV_PERCENTAGE)):]:
        test_df = pd.concat([test_df, i], ignore_index=True)

    train_df.to_csv(f'{out_dir}train.csv', index=False)
    dev_df.to_csv(f'{out_dir}dev.csv', index=False)
    test_df.to_csv(f'{out_dir}test.csv', index=False)


def split_window(window_size, input, target):
    """
    Split input and target to windows.
    """
    history = []
    future = []
    for i in range(window_size, len(input)):
        history.append(input[i - window_size:i])
        future.append(target[i])
    return history, future


def prepare_training_data(data_source, window_size, batch_size, shuffle):
    """
    Generate input for RCNN model.
    """
    if isinstance(data_source, str):
        data = pd.read_csv(data_source)
    else:
        data = data_source

    input_cols = COLUMNS[2:5]
    output_cols = COLUMNS[2:4]

    grouped = data.groupby([COLUMNS[0]])
    input_all = []
    target_all = []

    for g in grouped:
        g = g[1].sort_values(by=COLUMNS[1])
        history, future = split_window(window_size, g[input_cols].values, g[output_cols].values)
        input_all.extend(history)
        target_all.extend(future)
    input_all = np.array(input_all)
    target_all = np.array(target_all)
    res = tf.data.Dataset.from_tensor_slices((input_all, target_all))
    if shuffle:
        res = res.shuffle(input_all.shape[0]).batch(batch_size).repeat()
    else:
        res = res.batch(batch_size).repeat()
    return res


if __name__ == "__main__":

    data_sampling('./../data/200/')
