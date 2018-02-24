#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os

import numpy as np
import tensorflow as tf
import model.LineBased as LineBased
import data.dataset as dataset
from utils import parallel
from utils import hooks

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="_",
        usage="train.py [<args>] [-h | --help]")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        thickness=0.15,
        crop_skel=1.0,
        neighbor=5,
        device_list=[1,5],
        learning_rate_decay="noam",
        warmup_steps=2000,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        train_steps=10000,

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

        batch_size=10,
        thread_num=10,
        epoch=10,
        keep_top_checkpoint_max=5,
        eval_secs=None,
        eval_steps=100,

    )
    return params


def override_parameters(params, args):
    params.output = args.output or params.output
    params.parse(args.parameters)
    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay == "noam":
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = 1000 ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    params = default_parameters()
    override_parameters(params, args)


    # Build Graph
    with tf.Graph().as_default():
        dataset.start_queue(params)
        features = dataset.get_train_input(params)

        # Build model
        initializer = get_initializer(params)
        model = LineBased.Model(params)

        # Multi-GPU setting
        sharded_losses = parallel.parallel_model(
            model.get_training_func(initializer),
            features,
            params.device_list
        )
        loss = tf.add_n(sharded_losses) / len(sharded_losses)

        # Create global step
        global_step = tf.train.get_or_create_global_step()

        # Print parameters
        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0

        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = np.prod(np.array(v.shape.as_list())).tolist()
            total_size += v_size
        tf.logging.info("Total trainable variables size: %d", total_size)

        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)

        print('create opt')
        # Create optimizer
        opt = tf.train.AdamOptimizer(learning_rate,
                                     beta1=params.adam_beta1,
                                     beta2=params.adam_beta2,
                                     epsilon=params.adam_epsilon)

        train_op = tf.contrib.layers.optimize_loss(
            name="training",
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            clip_gradients=params.clip_grad_norm or None,
            optimizer=opt,
            colocate_gradients_with_ops=True
        )

        print('create hooks')
        # Add hooks
        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "loss": loss,
                },
                every_n_iter=1
            ),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=params.output,
                save_secs=params.save_checkpoint_secs or None,
                save_steps=params.save_checkpoint_steps or None,
                saver=tf.train.Saver(
                    max_to_keep=params.keep_checkpoint_max,
                    sharded=False
                )
            )
        ]

        config = session_config(params)

        train_hooks.append(
            hooks.EvaluationHook(
                model.get_evaluation_func(),
                dataset.get_eval_input,
                params.output,
                config,
                params.keep_top_checkpoint_max,
                eval_secs=params.eval_secs,
                eval_steps=params.eval_steps
            ))

        print('create session')
        # Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            while not sess.should_stop():
                sess.run(train_op)
            # coord.request_stop()
            # coord.join(threads)


if __name__ == "__main__":
    main(parse_args())
