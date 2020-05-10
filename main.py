import tensorflow as tf
from tensorflow.python.platform import app
import numpy as np

from opt_utils import opt, autoflags
from nets.disp_net import DispNet
from monodepth_dataloader_v3 import batch_from_dataset

EPOCHS = 100

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

def main(unused_argv):
    autoflags()
    if opt.trace == "":
        raise ValueError("OUT_DIR must be specified")

    ds_trn = batch_from_dataset()
    model = DispNet()
    model.compile(optimizer=tf.keras.optimizers.Adam())

    callbacks = []
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))
    model.fit(x=ds_trn, steps_per_epoch=10000, epochs=EPOCHS, verbose=1, callbacks=callbacks)

if __name__ == '__main__':
    app.run()
