import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
import numpy as np
import os

from opt_helper import opt, autoflags
from nets.disp_net import DispNet
from dataloader import batch_from_dataset

STEPS_PER_EPOCH = 10000
EPOCHS = opt.num_iterations // STEPS_PER_EPOCH

def lr_scheduler(epoch):
    lr = opt.learning_rate
    if isinstance(lr, (float, int)):
        return float(lr)
    elif isinstance(lr, (list, tuple)):
        t = epoch / (EPOCHS - 1)
        lr_max, lr_min = map(float, lr)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(t * np.pi))
    else:
        raise ValueError('Invalid learning rate')

if __name__ == '__main__':
    autoflags()
    if opt.trace == "":
        raise ValueError("OUT_DIR must be specified")

    strategy = tf.distribute.MirroredStrategy()
    print('* Number of devices: ', strategy.num_replicas_in_sync)
    with strategy.scope():
        model = DispNet()
        model.compile(optimizer=tf.keras.optimizers.Adam())

    callbacks = []
    callbacks.append(LearningRateScheduler(lr_scheduler))
    callbacks.append(ModelCheckpoint(os.path.join(opt.trace, 'weights-{epoch:03d}'), save_weights_only=True, save_best_only=False))
    callbacks.append(TensorBoard(log_dir=opt.trace, update_freq=100))

    ds_trn = batch_from_dataset()
    model.fit(x=ds_trn, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, callbacks=callbacks)
