import tensorflow as tf
import numpy as np
from tqdm import trange
import sys
import functools

from nets.disp_net import DispNet
from monodepth_dataloader_v3 import batch_from_dataset

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 20000 # 2500

# How often to save a model checkpoint
SAVE_INTERVAL = 2500

# Cosine annealing LR Scheduler
def lr_scheduler(lr, t):
    if isinstance(lr, (float, int)):
        return float(lr)
    elif isinstance(lr, (list, tuple)):
        lr_max, lr_min = map(float, lr)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(t * np.pi))
    else:
        raise ValueError('Invalid learning rate')

def train(opt):
    model = DispNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_step(imgL, imgR, dispL, dispR):
        with tf.GradientTape() as tape:
            _ = model([imgL, imgR, dispL, dispR], True)
            loss = sum(model.losses)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #loss_metric(loss)

    # lr_func = functools.partial(lr_scheduler, opt.learning_rate)
    # for itr in trange(start_itr, opt.num_iterations, file=sys.stdout):
    #     fetches = dict(train=train_op)
    #     if (itr) % (SUMMARY_INTERVAL) == 2:
    #         fetches['summary'] = summary_op
    #     outputs = sess.run(fetches, feed_dict={lr: lr_func(itr / opt.num_iterations)})

    # Iterate over the batches of the dataset.
    ds_iter = batch_from_dataset()
    for _ in trange(0, opt.num_iterations, file=sys.stdout):
        imageL, imageR, dispL, dispR = next(ds_iter)
        train_step(imageL, imageR, dispL, dispR)

    print('*** Done')
