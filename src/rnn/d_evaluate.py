import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

COLUMNS = ['episode', 'step', 'pos', 'vel', 'action']

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def d_evaluate(data_dir, out_dir, model, input_width, min_steps, train_n_episode=False, n_episode=5):

    df = pd.read_csv(data_dir)
    if train_n_episode is False:
        # Load episode with max. steps
        len_epi = df[COLUMNS[0]].max() + 1 - df[COLUMNS[0]].min()
        for ep in range(df[COLUMNS[0]].min(), df[COLUMNS[0]].max() + 1):
            if df[df[COLUMNS[0]] == ep][COLUMNS[1]].max() < min_steps:
                df = df.drop(df[df[COLUMNS[0]] == ep].index)
                len_epi -= 1
            else:
                pass
    else:
        len_epi = n_episode

    multi_evaluate = []
    pos_error_cal = []
    vel_error_cal = []
    for j in range(len_epi):
        row_max_steps = df.loc[df[COLUMNS[1]].idxmax()]
        df_test = df[df[COLUMNS[0]] == row_max_steps.episode]

        # define window behaviour
        state_buffer = collections.deque(maxlen=input_width)
        y_true = df_test[['pos', 'vel']].values
        y_predict = []
        for i in range(len(df_test)):
            if i < input_width:
                state_data = np.float64(
                    [df_test['pos'].values[i], df_test['vel'].values[i], df_test['action'].values[i]])
                state_buffer.append(state_data)
            else:
                state = np.array([list(state_buffer)])
                current = model.predict(np.float64(state))[0]
                y_predict.append(current)
                state_buffer.append(np.float64([current[0], current[1], df_test['action'].values[i]]))

        y_true = np.array(y_true)
        y_predict = np.array(y_predict)
        m = tf.keras.metrics.MeanAbsoluteError()
        m.update_state(y_true[:, 0][input_width:], y_predict[:, 0])
        pos_error = m.result().numpy()
        logger.info('Eval-MAE of position = %f' % pos_error)
        m.update_state(y_true[:, 1][input_width:], y_predict[:, 1])
        vel_error = m.result().numpy()
        logger.info('Eval-MAE of velocity = %f' % vel_error)
        pos_error_cal.append(pos_error)
        vel_error_cal.append(vel_error)
        df = df[~df[COLUMNS[0]].isin([row_max_steps.episode])]
        dic = {'pos_error': pos_error, 'vel_error': vel_error, 'y_true': y_true, 'y_predict': y_predict,
               'action': np.array(df_test['action'].values)}
        multi_evaluate.append(dic)

    # Mean, standard deviation, variance
    dic_cal = {'ave_pos_error': np.mean(pos_error_cal), 'var_pos_error': np.var(pos_error_cal),
               'arr_pos_error': np.std(pos_error_cal), 'ave_vel_error': np.mean(vel_error_cal),
               'var_vel_error': np.var(vel_error_cal), 'arr_vel_error': np.std(vel_error_cal)}
    cal_df = pd.DataFrame.from_dict(dic_cal, orient='index')
    cal_df.to_csv(f'{out_dir}/Eva_cal.csv')

    return multi_evaluate


def plotting_eval(multi_evaluate, window_size, out_dir=None):
    # TODO: move to notebook
    eval_para = ['pos_error', 'vel_error']
    for z in range(len(eval_para)):
        ep = eval_para[z]
        for k in range(2):
            multi_evaluate.sort(key=lambda j: j[ep])
            truth = multi_evaluate[-k].get('y_true')
            predictions = multi_evaluate[-k].get('y_predict')
            actions = multi_evaluate[-k].get('action')
            plt.figure(figsize=(20, 20))
            fields = COLUMNS[2:4]
            fig, axs = plt.subplots(len(fields) + 1, 1)
            for i in range(len(fields)):
                f = fields[i]
                axs[i].plot(range(len(predictions)), truth[:, i][window_size:], label=fields[i])
                axs[i].plot(range(len(predictions)), predictions[:, i], label='prediction', ls='--')
                axs[i].grid()
                axs[i].legend(loc='best')
            axs[len(fields)].plot(range(len(predictions)), actions[window_size:], label='action')
            axs[len(fields)].grid()
            axs[len(fields)].legend(loc='best')
            plt.suptitle('pos_error: %s, vel_error:%s' % (
                multi_evaluate[-k].get('pos_error'), multi_evaluate[-k].get('vel_error')))
            if out_dir:
                if ep == 'pos_error' and k == 0:
                    plt.savefig(f'{out_dir}/best_pos.png')
                elif ep == 'vel_error' and k == 0:
                    plt.savefig(f'{out_dir}/best_vel.png')
                elif ep == 'pos_error' and k == 1:
                    plt.savefig(f'{out_dir}/worst_pos.png')
                elif ep == 'vel_error' and k == 1:
                    plt.savefig(f'{out_dir}/worst_vel.png')
            else:
                plt.show()


