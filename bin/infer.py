#!/usr/bin/env python
# coding=utf-8

import argparse
import os

import tensorflow as tf

import model.pixellink as pixellink
import data.dataset as dataset
import utils.parallel as parallel
import postprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infering images using existing models",
        usage="infer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    return parser.parse_args()


def default_parameters():
    # params = tf.contrib.training.HParams()
    params = tf.contrib.training.HParams(
        thickness=0.15,
        crop_skel=1.0,
        neighbor=5,
        device_list=[3],
        warmup_steps=2000,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        train_steps=100000,

        initializer="uniform",
        clip_grad_norm=5.0,
        output='/home/rjq/train',
        save_checkpoint_steps=100,
        keep_checkpoint_max=5,
        initializer_gain=0.08,
        learning_rate=0.01,
        save_checkpoint_secs=None,
        basenets='vgg16',
        input_size=[512,512,3],
        Label_size=[512,512,5],
        padding='SAME',
        pooling='max',
        basenet='vgg16',
        upsampling='DeConv',
        US_Params='3 3 2 2 same ReLU',
        Predict_stage=4,
        predict_channels=[128, 64, 32, 32],

        batch_size=32,
        thread_num=10,
        keep_top_checkpoint_max=5,
        eval_secs=None,
        eval_steps=10,
        prefetch_buffer=500,
        shuffle_buffer=100,
        epoch=400,

        # pixellink
        weight_decay=0.0005,
        momentum=0.9,
        optimizer='sgd_momentum',
        learning_rate_decay="pixellink",

        output_scalar=2,
    )

    return params


def import_params(model_dir, params):

    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, "params.json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)
    return params



def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]

        if batch < num_shards:
            feed_dict[placeholders[0][name]] = feat
            n = 1
        else:
            shard_size = (batch + num_shards - 1) // num_shards

            for i in range(num_shards):
                shard_feat = feat[i * shard_size:(i + 1) * shard_size]
                feed_dict[placeholders[i][name]] = shard_feat
                n = num_shards

    return predictions[:n], feed_dict


def reconstruct(img, maps):
    processor = postprocessing.Postprocessor()
    # np.save('maps.npy', maps)
    print("-----------process started-----------")
    ctns = processor.process(maps)
    print("-----------process ended-------------")
    # os.system('rm maps.npy')
    return ctns


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    params_list = [default_parameters() for _ in range(len(args.checkpoints))]

    params_list = [
        import_params(args.checkpoints[i], params_list[i])
        for i in range(len(args.checkpoints))
    ]

    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(args.checkpoints))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i, checkpoint in enumerate(args.checkpoints):
            tf.logging.info("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)

        # Build models
        model_fns = []

        for i in range(len(args.checkpoints)):
            model = pixellink.PixelLinkNetwork(params_list[i], 'PixelLinkNetwork' + "_%d" % i)
            model_fn = model.get_inference_func()
            model_fns.append(model_fn)

        params = params_list[0]
        # Build input queue
        features = dataset.get_inference_input(params)
        # Create placeholders
        placeholders = []

        for i in range(len(params.device_list)):
            placeholders.append({
                "input_img": tf.placeholder(tf.float32, [None, None, None, 3],
                                            "input_img_%d" % i),
                'lens': tf.placeholder(tf.float32,[None,], 'lens_%d'%i),
                'cnts': tf.placeholder(tf.float32, [None,None, None,None], 'cnts_%d'%i),
                'care': tf.placeholder(tf.float32,[None,], 'care_%d'%i),

            })
            # {'input_img': (
            #     tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
            #     3),
            # 'lens': (tf.Dimension(None),),
            # 'cnts': (
            #     tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
            #     tf.Dimension(None)),
            # 'care': (tf.Dimension(None),)}
            # )


            # A list of outputs
        predictions = parallel.data_parallelism(
            params.device_list,
            model_fns,
            placeholders)

        # Create assign ops
        assign_ops = []

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []

            for v in all_var_list:
                if v.name.startswith('PixelLinkNetwork' + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                'PixelLinkNetwork' + "_%d" % i)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)
        results = []

        # Create session
        with tf.Session(config=session_config(params)) as sess:
            # Restore variables
            sess.run(assign_op)
            sess.run(tf.tables_initializer())

            while True:
                try:
                    feats = sess.run(features)
                    op, feed_dict = shard_features(feats, placeholders,
                                                   predictions)
                    temp = sess.run(predictions, feed_dict=feed_dict)
                    print(temp)
                    results.append(temp)
                    message = "Finished batch %d" % len(results)
                    tf.logging.log(tf.logging.INFO, message)
                    #TODO: save and reconstruct
                    for res in results:
                        print(len(res))
                        print(res)
                        print(res.shape)

                    for i in range(len(predictions)):
                        res = reconstruct(None, predictions[i])
                        print(res)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":
    main(parse_args())
