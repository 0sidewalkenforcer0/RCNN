import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing.data_preprocessing import MountainCarNew
import seaborn as sns

C_MODEL_COLUMNS = ['num_model', 'episode', 'step', 'pos', 'vel', 'action', 'reward']
REPORT_COLUMNS = ['train_size', 'num_model', 'episode', 'steps', 'reward', 'done']


def c_evaluate(models, out_dir, steps, episodes, min_pos=-0.6, max_pos=-0.4):
    """
    Evaluate RCNN.
    :param models: A list of trained models.
    :param out_dir: Path to save the .csv file with the C_MODEL_COLUMNS
    :param steps: Largest iterations each episode.
    :param episodes: Number of episodes.
    :param min_pos: Lowest initial position.
    :param max_pos: Highest initial position.
    :return:
    """
    env = MountainCarNew()
    env.np_random.seed(0)
    samples = []
    for num_model in range(len(models)):
        model = models[num_model]
        episode_steps = []
        state_buffer = collections.deque(maxlen=model.window_size)
        for episode in range(episodes):

            # initialize each episode
            env.reset_nn(model.window_size, state_buffer, min_pos, max_pos)
            step = 0
            done = False

            # write first window
            for ob in list(state_buffer):
                step += 1
                samples.append({C_MODEL_COLUMNS[0]: num_model + 1,
                                C_MODEL_COLUMNS[1]: episode + 1,
                                C_MODEL_COLUMNS[2]: step,
                                C_MODEL_COLUMNS[3]: ob[0],
                                C_MODEL_COLUMNS[4]: ob[1],
                                C_MODEL_COLUMNS[5]: ob[2],
                                C_MODEL_COLUMNS[6]: 0})

            while step < steps and not done:
                step += 1

                # model prediction
                model_input = np.array([list(state_buffer)])
                action, next_state, next_action, rewards = model(np.float64(model_input))
                pos = np.float(next_state[0][0])
                vel = np.float(next_state[0][1])
                # constraints
                if np.around(pos, 3) == env.min_position and vel < 0:
                    vel = 0
                action = np.float(action[0])
                next_action = np.float(next_action[0])
                reward = np.float(rewards[0])

                # replace last action
                samples[-1][C_MODEL_COLUMNS[5]] = action
                # write output
                samples.append({C_MODEL_COLUMNS[0]: num_model + 1,
                                C_MODEL_COLUMNS[1]: episode + 1,
                                C_MODEL_COLUMNS[2]: step,
                                C_MODEL_COLUMNS[3]: pos,
                                C_MODEL_COLUMNS[4]: vel,
                                C_MODEL_COLUMNS[5]: next_action,
                                C_MODEL_COLUMNS[6]: reward})
                state_buffer.append(np.float64([pos, vel, action]))

                done = bool(
                    pos >= env.goal_position and vel >= env.goal_velocity
                )

            episode_steps.append(step)
            print('current episode %d ends in %d steps' % (episode, step))

    env.close()
    df = pd.DataFrame(samples, columns=C_MODEL_COLUMNS)
    df.to_csv(out_dir, index=False)


def c_report(models, size, out_dir, steps, episodes):
    env = MountainCarNew()
    env.np_random.seed(0)
    samples = []
    for num_model in range(len(models)):
        model = models[num_model]
        state_buffer = collections.deque(maxlen=model.window_size)

        for episode in range(episodes):

            # initialize each episode
            env.reset_nn(model.window_size, state_buffer)
            step = model.window_size
            done = False
            rewards = []

            while step < steps and not done:
                step += 1

                # model prediction
                model_input = np.array([list(state_buffer)])
                action, next_state, next_action, reward = model(np.float64(model_input))
                pos = np.float(next_state[0][0])
                vel = np.float(next_state[0][1])
                # constraints
                if np.around(pos, 3) == env.min_position and vel < 0:
                    vel = 0
                action = np.float(action[0])
                # next_action = np.float(next_action[0])
                reward = np.float(reward[0])

                # accumulate reward
                rewards.append(reward)

                done = bool(
                    pos >= env.goal_position and vel >= env.goal_velocity
                )
                state_buffer.append(np.float64([pos, vel, action]))

            # write output
            samples.append({REPORT_COLUMNS[0]: size,
                            REPORT_COLUMNS[1]: num_model + 1,
                            REPORT_COLUMNS[2]: episodes,
                            REPORT_COLUMNS[3]: step,
                            REPORT_COLUMNS[4]: np.sum(rewards),
                            REPORT_COLUMNS[5]: done})
            print('current episode %d ends in %d steps' % (episode, step))

    env.close()
    df = pd.DataFrame(samples, columns=REPORT_COLUMNS)
    df.to_csv(out_dir, index=False)


def c_plot_all(data_source, out_dir):
    df = pd.read_csv(data_source)
    df.plot(subplots=True, figsize=(10, 15), grid=True)
    plt.savefig(out_dir)


def c_plot_dynamic(data_source, out_dir):
    df = pd.read_csv(data_source)
    grouped = df.groupby(C_MODEL_COLUMNS[1])
    steps = list(grouped[C_MODEL_COLUMNS[2]].max())
    sns.set_style('darkgrid')
    sns.scatterplot(data=df, x=C_MODEL_COLUMNS[3], y=C_MODEL_COLUMNS[4], hue=C_MODEL_COLUMNS[1],
                    markers=True, legend=False)
    plt.savefig(out_dir)

    return steps


def c_plot_report(data_dir, sizes, output_dir):
    df = pd.DataFrame()
    for size in sizes:
        df_current = pd.read_csv(f'{data_dir}m{size}/results.csv')
        df = pd.concat([df, df_current], ignore_index=True)
    grouped = df.groupby([REPORT_COLUMNS[0], REPORT_COLUMNS[1]])
    steps = []
    sizes = []
    rates = []
    for name, group in grouped:

        done = group[group[REPORT_COLUMNS[5]]]
        step = done[REPORT_COLUMNS[3]].mean()
        rate = len(done) / len(group)
        rates.append(rate)
        steps.append(step)
        sizes.append(name[0])

    samples = {'size': sizes, 'step': steps, 'rate': rates}

    sns.set_style('darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    sns.lineplot(x=samples['size'], y=samples['rate'], ax=axes[0])
    sns.lineplot(x=samples['size'], y=samples['step'], ax=axes[1])

    axes[0].set_ylabel('success rate')
    axes[1].set_ylabel('avg. steps')

    plt.subplots_adjust(wspace=0.3, hspace=0)
    fig.text(0.5, 0.05, 'window size', ha='center')
    fig.savefig(output_dir)

    return rates


if __name__ == "__main__":
    pass
