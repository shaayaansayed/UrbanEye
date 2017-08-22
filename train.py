from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
from scipy.misc import imread
import os
import utils 
import data_loader 

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("tr_nbatch", 2, "batch size for training")
tf.flags.DEFINE_integer("val_nbatch", 2, "batch size for training")
tf.flags.DEFINE_string("ckpt_dir", "", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/scratch/cluster/vsub/ssayed/nga/dataset/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("print_every", 1, "iterations between print summary")
tf.flags.DEFINE_integer("decay_after", 15, "epochs to decay after")
tf.flags.DEFINE_float("decay_rate", 0.98, "epochs to decay after")
tf.flags.DEFINE_float("decay_every", 1, "decay every one epoch")
tf.flags.DEFINE_integer("log_tr_img_every", 1, "iterations between train image summary logging")
tf.flags.DEFINE_integer("log_val_img_every", 10, "iterations between train image summary logging")
tf.flags.DEFINE_integer("save_model_every", 5, "number of epochs for model save")
tf.flags.DEFINE_string("model_dir", '/scratch/cluster/ssayed/nga/Model_zoo', "Path to vgg model mat")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 9
IMAGE_SIZE = 224

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return annotation_pred, conv_t3


def train(lr, loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(lr)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    kp_pl = tf.placeholder(tf.float32, name="keep_probabilty")
    input_pl = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    target_pl = tf.placeholder(tf.int64, shape=[None, IMAGE_SIZE, IMAGE_SIZE], name="annotation")

    pred_op, logits_op = inference(input_pl, kp_pl)
    loss_op = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_op,
                                                                          labels=target_pl,
                                                                          name="entropy")))
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, target_pl), tf.float32))
    tr_vars = tf.trainable_variables()
    train_op = train(FLAGS.learning_rate, loss_op, tr_vars)

    global_step_pl = tf.placeholder(tf.float32, name='global_step')
    lr_decay = tf.train.exponential_decay(FLAGS.learning_rate, global_step_pl, FLAGS.decay_every, FLAGS.decay_rate)
    train_op_decay = train(lr_decay, loss_op, tr_vars) 

    img_summary_pl = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE*2, IMAGE_SIZE*2, 3], name='img_summary_pl')
    tr_img_summary_op = tf.summary.image('train_image', img_summary_pl, max_outputs=FLAGS.tr_nbatch)
    val_img_summary_op = tf.summary.image('val_image', img_summary_pl, max_outputs=FLAGS.val_nbatch)

    metrics = ['loss', 'accuracy']
    metric_pl = [tf.placeholder(tf.float32, [], '{}_pl'.format(metric_name)) for metric_name in metrics]
    with tf.name_scope('train_metrics') :
        tr_metrics = [tf.summary.scalar(name, pl) for name, pl in zip(metrics, metric_pl)]
        tr_metrics_op = tf.summary.merge(tr_metrics)

    with tf.name_scope('val_metrics') :
        val_metrics = [tf.summary.scalar(name, pl) for name, pl in zip(metrics, metric_pl)]
        val_metrics_op = tf.summary.merge(val_metrics)

    remove = glob.glob(os.path.join(FLAGS.ckpt_dir, '*'))
    for f in remove :
        os.remove(f)

    dl = data_loader.DataLoader(FLAGS.data_dir, 224, 0.15)

    saver = tf.train.Saver(max_to_keep=100)

    tr_ix = 0
    val_ix = 0
    epoch_ix = 0
    best_val_loss = 1e10
    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        fw = tf.summary.FileWriter(FLAGS.ckpt_dir, graph=sess.graph)

        while True :

            inputs, labels, wrap = dl.next_train_batch(FLAGS.tr_nbatch)

            if epoch_ix < FLAGS.decay_after :
                pred, loss, accuracy, _ = sess.run([pred_op, loss_op, accuracy_op, train_op], \
                                                    {input_pl:inputs, target_pl:labels, kp_pl:0.85})
            else :
                pred, loss, accuracy, _ = sess.run([pred_op, loss_op, accuracy_op, train_op_decay], \
                                                    {input_pl:inputs, target_pl:labels, kp_pl:0.85, global_step_pl:epoch_ix-FLAGS.decay_after})

            tr_metrics_pb = sess.run(tr_metrics_op, {pl:val for pl,val in zip(metric_pl, [loss, accuracy])})
            fw.add_summary(tr_metrics_pb, tr_ix)

            if tr_ix % FLAGS.log_tr_img_every == 0 :
                tr_img_summary = utils.create_img_summary(inputs, labels, pred)
                tr_img_pb = sess.run(tr_img_summary_op, {img_summary_pl: tr_img_summary})
                fw.add_summary(tr_img_pb, tr_ix)

            if wrap :
                val_wrap = False
                losses = []
                accuracies = []
                while not val_wrap :
                    val_inputs, val_labels, val_wrap, _ = dl.next_val_batch(FLAGS.val_nbatch)
                    val_pred, val_loss, val_accuracy = sess.run([pred_op, loss_op, accuracy_op], {input_pl:val_inputs, target_pl:val_labels, kp_pl:1.0})

                    if val_ix % FLAGS.log_val_img_every == 0 :
                        val_img_summary = utils.create_img_summary(val_inputs, val_labels, val_pred)
                        val_img_pb = sess.run(val_img_summary_op, {img_summary_pl: val_img_summary})
                        fw.add_summary(val_img_pb, val_ix)

                    losses.append(val_loss)
                    accuracies.append(val_accuracy)
                    val_ix += 1

                total_val_loss = np.mean(losses)
                total_val_accuracy = np.mean(accuracy)

                val_metrics_pb = sess.run(val_metrics_op, {pl:val for pl,val in zip(metric_pl, [total_val_loss, total_val_accuracy])})
                fw.add_summary(val_metrics_pb, epoch_ix)

                if epoch_ix % FLAGS.save_model_every == 0 :
                    saver.save(sess, os.path.join(FLAGS.ckpt_dir, 'model'), epoch_ix)

                if total_val_loss < best_val_loss :
                    best_val_loss = total_val_loss
                    saver.save(sess, os.path.join(FLAGS.ckpt_dir, 'best_val_loss'))

                print('epoch {}: loss - {:.3f}, accuracy - {:.3f}'.format(epoch_ix, total_val_loss, total_val_accuracy))
                epoch_ix = epoch_ix + 1 if wrap else epoch_ix

            tr_ix += 1
            fw.flush()


main()