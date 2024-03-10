import tensorflow as tf
from data_preprocessing.data_preprocessing import prepare_training_data, get_normalizations
from rnn.d_rnn import DModel
from rcnn.c_rnn import CModel


tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_v2_behavior()

DATA_PATH = '../data/'
COLUMNS = ['episode', 'step', 'pos', 'vel', 'action']
WINDOW_SIZE = 8
BATCH_SIZE = 32
RNN_PATH = '../models/rnn/'
RCNN_PATH = '../models/rcnn/'


def compile_and_fit(model,
                    output_dir,
                    train_data,
                    dev_data,
                    delta=1e-6,
                    learning_rate=0.001,
                    patience_lr=10,
                    loss='mse',
                    epochs=500,
                    steps_per_epoch=100,
                    validation_steps=100,
                    validation_freq=1,
                    patience_gradient=20,
                    metrics=None,
                    optimizer=tf.optimizers.Adam):
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.1,
                                                       patience=patience_lr,
                                                       verbose=1,
                                                       mode='min',
                                                       min_delta=delta,
                                                       cooldown=0,
                                                       min_lr=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=patience_gradient,
                                                      min_delta=delta,
                                                      restore_best_weights=True,
                                                      mode='min',
                                                      verbose=1)
    train_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{output_dir}bestTrainLoss/',
                                                        monitor='loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='min')
    dev_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{output_dir}bestValLoss/',
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      mode='min')

    model.compile(loss=loss,
                  run_eagerly=True,
                  optimizer=optimizer(learning_rate=learning_rate),
                  metrics=metrics)

    model.fit(train_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=dev_data,
              validation_steps=validation_steps,
              validation_freq=validation_freq,
              callbacks=[early_stopping, train_callback, dev_callback, lr_callback])


if __name__ == '__main__':

    # load data
    train = prepare_training_data(data_source=f'{DATA_PATH}200/train.csv',
                                  window_size=WINDOW_SIZE,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    dev = prepare_training_data(data_source=f'{DATA_PATH}200/dev.csv',
                                window_size=WINDOW_SIZE,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    mean_in, std_in = get_normalizations(COLUMNS[2:5], f'{DATA_PATH}200/train.csv')
    mean_out, std_out = get_normalizations(COLUMNS[2:4], f'{DATA_PATH}200/train.csv')
    mean_a, std_a = get_normalizations(COLUMNS[4], f'{DATA_PATH}200/train.csv')

    # train d model
    rnn = DModel(mean_in, std_in, mean_out, std_out)
    d_metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    compile_and_fit(rnn,
                    f'{RNN_PATH}/test/',
                    train,
                    dev,
                    metrics=d_metrics)

    del rnn

    # train c model
    rnn = DModel(mean_in, std_in, mean_out, std_out)
    rnn.load_weights(f'{RNN_PATH}test/bestValLoss/')
    rnn.trainable = False
    rcnn = CModel(rnn, WINDOW_SIZE, mean_in, std_in, mean_out, std_out, mean_a, std_a)
    compile_and_fit(rcnn,
                    f'{RCNN_PATH}test/',
                    train,
                    dev,
                    loss=rcnn.losses)

