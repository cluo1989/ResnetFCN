import tensorflow as tf 
import numpy as np 
import resnet_fcn
import time
import os
from data.dataset import create_dataset

PATH = "./pretraineds/"
META_FN = PATH + "ResNet-L50.meta"
CHECKPOINT_FN = PATH + "ResNet-L50.ckpt"

MOMENTUM = 0.9
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', './data', "Directory where to store dataset.")
tf.app.flags.DEFINE_string('train_dir', './logs', "Directory where to write event logs and checkpoint")
tf.app.flags.DEFINE_float('learning_rate', 0.0001, "learning rate")
tf.app.flags.DEFINE_integer('batch_size', 5, "batch size")

NUM_CLASSES = 11
IMAGE_WIDTH = 224 #256
IMAGE_HEIGHT = 224 #384

params = {'num_parallel_calls':1, 
          'batch_size': FLAGS.batch_size, 
          'image_width': IMAGE_WIDTH, 
          'image_height': IMAGE_HEIGHT,
          'num_epochs': 200, 
          }


if __name__ == '__main__':
    sess = tf.Session()
    batch_size = FLAGS.batch_size

    # dataset
    data_dir = FLAGS.data_dir
    train_data_dir_images = os.path.join(data_dir, "train/images/")
    train_data_dir_labels = os.path.join(data_dir, "train/labels/")
    dev_data_dir = os.path.join(data_dir, "dev")

    # Get the filenames from the trainsets
    image_names = [os.path.join(train_data_dir_images, f) for f in os.listdir(train_data_dir_images) if f.endswith('.png')]
    label_names = [os.path.join(train_data_dir_labels, f) for f in os.listdir(train_data_dir_labels) if f.endswith('.png')]

    # label_names = [int(f.split('/')[-1][0]) for f in train_filenames]
    params['train_size'] = len(image_names)

    inputs = create_dataset(True, image_names, label_names, params)
    print('labels shape:', inputs['labels'].get_shape())

    # build graph
    # x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    # labels = tf.placeholder(tf.int32, [batch_size, 224, 224])
    pred, logits = resnet_fcn.inference(inputs['images'], is_training=True, num_classes=NUM_CLASSES, num_blocks=[3,4,6,3])
    tf.summary.image('pred', tf.cast(pred, tf.uint8))
    print('logits shape:', logits.get_shape())

    # compute loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=inputs['labels'], name="entropy")))
    tf.summary.scalar('loss', loss)

    # get optimizer
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)

    # compute and apply grads
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # group train and bn operation
    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    # tf.train.start_queue_runners(sess=sess)

    # Init global variables
    sess.run(tf.global_variables_initializer())

    # Restore variables
    # global_step not found in ckpt, tf.global_variables() -> tf.trainable_variables()
    saver = tf.train.Saver([var for var in tf.trainable_variables() if "scale_fcn" not in var.name])
    restore_variables = True
    if restore_variables:
        saver.restore(sess, CHECKPOINT_FN)

    # new saver
    saver = tf.train.Saver(tf.global_variables())
    sum_op = tf.summary.merge_all()  # merge([loss_summary, ...])
    writer = tf.summary.FileWriter('./logs/', sess.graph)

    for epoch in range(params['num_epochs']):
        num_steps = (params['train_size']+params['batch_size']-1)//params['batch_size']
        sess.run(inputs['iterator_init_op'])
        for _ in range(num_steps):
            start_time = time.time()
            step = sess.run(global_step)  # get global step

            # run train_op and get loss
            run_op = [train_op, loss, sum_op]
            o = sess.run(run_op)

            loss_value = o[1]
            duration = time.time() - start_time

            # print
            if step%5 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('Epoch: %d, step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                writer.add_summary(o[2], step)
                print(format_str % (epoch, step, loss_value, examples_per_sec, duration))

            # save
            if step > 1 and step % 500 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model%d.ckpt' % step)
                saver.save(sess, checkpoint_path, global_step=global_step)