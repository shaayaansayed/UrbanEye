from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
from scipy.misc import imread, imsave
import os
import utils 
import data_loader 

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("test_nbatch", 2, "batch size for training")
tf.flags.DEFINE_string("restore_from", "", "path to logs directory")
tf.flags.DEFINE_string("test_dir", "/scratch/cluster/vsub/ssayed/nga/dataset/", "path to dataset")
tf.flags.DEFINE_integer("log_test_img_every", 1, "iterations between train image summary logging")
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


def main(argv=None):
    kp_pl = tf.placeholder(tf.float32, name="keep_probabilty")
    input_pl = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")

    pred_op, logits_op = inference(input_pl, kp_pl)

    img_summary_pl = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE*2, 3], name='img_summary_pl')
    test_img_summary_op = tf.summary.image('test', img_summary_pl, max_outputs=FLAGS.test_nbatch)

    logdir = os.path.join(FLAGS.test_dir, 'logs')
    if not os.path.isdir(logdir) :
        os.mkdir(logdir)
    remove = glob.glob(os.path.join(logdir, '*'))
    for f in remove :
        os.remove(f)

    test_dir_base = os.path.basename(os.path.normpath(FLAGS.test_dir))
    output_dir = os.path.join(FLAGS.test_dir, '{}_outputs'.format(test_dir_base))
    if not os.path.isdir(output_dir) :
        os.mkdir(output_dir)

    dl = data_loader.DataLoader(FLAGS.test_dir, 224, -1)

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.restore_from)
        fw = tf.summary.FileWriter(logdir, graph=sess.graph)

        test_ix = 0
        wrap = False
        while not wrap :

            inputs, wrap, filepaths = dl.next_test_batch(FLAGS.test_nbatch)

            pred = sess.run(pred_op, {input_pl:inputs, kp_pl:1.0})

            test_img_summary = utils.create_test_img_summary(inputs, pred)
            if test_ix % FLAGS.log_test_img_every == 0 :
                test_img_pb = sess.run(test_img_summary_op, {img_summary_pl: test_img_summary})
                fw.add_summary(test_img_pb, test_ix)

            for ix, path in enumerate(filepaths) :
                filename = os.path.basename(path).split('.')[0]
                np.save(os.path.join(output_dir, filename), pred[ix, :, :])

            test_ix += 1
            fw.flush()


main()