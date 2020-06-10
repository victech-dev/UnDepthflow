import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
import numpy as np
import os

from opt_helper import opt, autoflags
from disp_net import DispNet
from dataloader import batch_from_dataset

STEPS_PER_EPOCH = 10000
EPOCHS = opt.num_iterations // STEPS_PER_EPOCH

if __name__ == '__main__':
    autoflags()
    if opt.trace == "":
        raise ValueError("OUT_DIR must be specified")

    strategy = tf.distribute.MirroredStrategy()
    print('* Number of devices: ', strategy.num_replicas_in_sync)
    with strategy.scope():
        disp_net = DispNet('train')

        if opt.pretrained_model:
            print('* Loading pretrained model:', opt.pretrained_model)
            disp_net.model.load_weights(opt.pretrained_model)

        # Cosine annealing lr if required
        if isinstance(opt.learning_rate, (list, tuple)):
            lr = list(map(float, opt.learning_rate))
            lr = tf.keras.experimental.CosineDecay(lr[0], opt.num_iterations, alpha=lr[1]/lr[0])
        else:
            lr = float(opt.learning_rate)

        disp_net.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        disp_net.model.summary()

    callbacks = []
    callbacks.append(ModelCheckpoint(os.path.join(opt.trace, 'weights-{epoch:03d}.h5'), save_weights_only=True, save_best_only=False))
    callbacks.append(TensorBoard(log_dir=opt.trace, update_freq=100))

    ds_trn = batch_from_dataset()
    disp_net.model.fit(ds_trn, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1, callbacks=callbacks)
