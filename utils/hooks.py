# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import operator
import os
import pickle

import tensorflow as tf
import numpy as np
from utils.evaluate import evaluate
import postprocessing
import cv2


def _get_saver():
    # Get saver from the SAVERS collection if present.
    collection_key = tf.GraphKeys.SAVERS
    savers = tf.get_collection(collection_key)

    if not savers:
        raise RuntimeError("No items in collection {}. "
                           "Please add a saver to the collection ")
    elif len(savers) > 1:
        raise RuntimeError("More than one item in collection")

    return savers[0]


def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _read_checkpoint_def(filename):
    records = []

    with tf.gfile.GFile(filename) as fd:
        fd.readline()

        for line in fd:
            records.append(line.strip().split(":")[-1].strip()[1:-1])

    return records


def _save_checkpoint_def(filename, checkpoint_names):
    keys = []

    for checkpoint_name in checkpoint_names:
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, checkpoint_name))

    sorted_names = sorted(keys, key=operator.itemgetter(0),
                          reverse=True)

    with tf.gfile.GFile(filename, "w") as fd:
        fd.write("model_checkpoint_path: \"%s\"\n" % checkpoint_names[0])

        for checkpoint_name in sorted_names:
            checkpoint_name = checkpoint_name[1]
            fd.write("all_model_checkpoint_paths: \"%s\"\n" % checkpoint_name)


def _read_score_record(filename):
    # "checkpoint_name": score
    records = []

    if not tf.gfile.Exists(filename):
        return records

    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with tf.gfile.GFile(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    # Sort
    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records


def _depad(cnts, lens):
    news = []
    for i in range(len(lens)):
        news.append(cnts[i][:lens[i], :, :])
    return news

def reconstruct(img, maps):
    processor = postprocessing.Postprocessor()
    # np.save('maps.npy', maps)
    print("-----------process started-----------")
    ctns = processor.process(maps)
    print("-----------process ended-------------")
    # os.system('rm maps.npy')
    return ctns

def _softmax(x):
    x = np.exp(x)
    return x[:, :, 1:2] / np.expand_dims(np.sum(x, axis=2), 2)

def _evaluate(eval_fn, input_fn, path, config, save_path):
    graph = tf.Graph()
    with graph.as_default():
        features = input_fn()
        prediction = eval_fn(features)
        results = {
            'prediction': prediction,
            'input_img': features['input_img'],
            'lens': features['lens'],
            'cnts': features['cnts'],
            'care': features['care']
        }
        sess_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=path,
            config=config
        )

        recall_sum, gt_n_sum, precise_sum, pred_n_sum = 0,0,0,0
        with tf.train.MonitoredSession(session_creator=sess_creator) as sess:
            tf.logging.info('start evaluation')
            time = 0
            while not sess.should_stop():
                time += 1
                print('time', time)
                outputs = sess.run(results)
                img = outputs['input_img']
                prediction = outputs['prediction']
                lens = outputs['lens']
                cnts = outputs['cnts']
                cnts = [(x/2).astype(np.int32) for x in cnts]
                cnts = _depad(cnts, lens)
                care = outputs['care']
                # imname = outputs['imname']
                # print(imname)
                for i in range(img.shape[0]):
                    re_cnts = reconstruct(img[i], prediction[i])
                    TR, TP, T_gt_n, T_pred_n, PR, PP, P_gt_n, P_pred_n = \
                        evaluate(img[i],cnts,re_cnts,care)
                    tf.logging.info(' recall: '+str(TR)+'; precise: '+str(TP))
                    recall_sum+=TR*T_gt_n
                    precise_sum+=TP*T_pred_n
                    gt_n_sum+=T_gt_n
                    pred_n_sum+=T_pred_n

                    height, width = prediction.shape[1], prediction.shape[2]
                    imgoutput = np.zeros(shape=(height*2, width*2, 3), dtype=np.uint8)
                    imgoutput[0:height, width:width*2, :] = cv2.resize(img[0], (width, height))
                    imgoutput[height:height*2, width:width*2, :] = (_softmax(prediction[i, :, :, 0:2])*255).astype(np.uint8)
                    cv2.drawContours(imgoutput, cnts, -1, (0, 0, 255))
                    cv2.drawContours(imgoutput, re_cnts, -1, (0, 255, 0))
                    cv2.imwrite(os.path.join(save_path,'output_{:03d}.png'.format(time)), imgoutput)

        if int(gt_n_sum) != 0:
            ave_r = recall_sum/gt_n_sum
        else:
            ave_r = 0.0
        if int(pred_n_sum) != 0:
            ave_p = precise_sum/pred_n_sum
        else:
            ave_p = 0.0
        if ave_r != 0.0 and ave_p != 0.0:
            ave_f = 2/(1/ave_r+1/ave_p)
        else:
            ave_f = 0.0
        tf.logging.info('ave recall:{}, precise:{}, f:{}'.format(ave_r,ave_p,ave_f))
        tf.logging.info('end evaluation')
        return ave_f


class EvaluationHook(tf.train.SessionRunHook):
    """ Validate and save checkpoints every N steps or seconds.
        This hook only saves checkpoint according to a specific metric.
    """

    def __init__(self, eval_fn, eval_input_fn, base_dir,
                 session_config, max_to_keep=5, eval_secs=None,
                 eval_steps=None, metric="f_score"):
        """ Initializes a `EvaluationHook`.
        :param eval_fn: A function with signature (feature)
        :param eval_input_fn: A function with signature ()
        :param base_dir: A string. Base directory for the checkpoint files.
        :param session_config: An instance of tf.ConfigProto
        :param max_to_keep: An integer. The maximum of checkpoints to save
        :param eval_secs: An integer, eval every N secs.
        :param eval_steps: An integer, eval every N steps.
        :param checkpoint_basename: `str`, base name for the checkpoint files.
        :raises ValueError: One of `save_steps` or `save_secs` should be set.
        :raises ValueError: At most one of saver or scaffold should be set.
        """
        tf.logging.info("Create EvaluationHook.")

        if metric != "f_score":
            raise ValueError("Currently, EvaluationHook only support f_score")

        self._base_dir = base_dir.rstrip("/")
        self._session_config = session_config
        self._save_path = os.path.join(base_dir, "eval")
        self._record_name = os.path.join(self._save_path, "record")
        self._log_name = os.path.join(self._save_path, "log")
        self._eval_fn = eval_fn
        self._eval_input_fn = eval_input_fn
        self._max_to_keep = max_to_keep
        self._metric = metric
        self._global_step = None
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=eval_secs or None, every_steps=eval_steps or None
        )

    def begin(self):
        if self._timer.last_triggered_step() is None:
            self._timer.update_last_triggered_step(0)

        global_step = tf.train.get_global_step()

        if not tf.gfile.Exists(self._save_path):
            tf.logging.info("Making dir: %s" % self._save_path)
            tf.gfile.MakeDirs(self._save_path)

        params_pattern = os.path.join(self._base_dir, "*.json")
        params_files = tf.gfile.Glob(params_pattern)

        for name in params_files:
            new_name = name.replace(self._base_dir, self._save_path)
            tf.gfile.Copy(name, new_name, overwrite=True)

        if global_step is None:
            raise RuntimeError("Global step should be created first")

        self._global_step = global_step

    def before_run(self, run_context):
        args = tf.train.SessionRunArgs(self._global_step)
        return args

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results

        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step)

            # Get the real value
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                # Save model
                save_path = os.path.join(self._base_dir, "model.ckpt")
                saver = _get_saver()
                tf.logging.info("Saving checkpoints for %d into %s." %
                                (global_step, save_path))
                saver.save(run_context.session,
                           save_path,
                           global_step=global_step)
                # Do validation here
                tf.logging.info("Validating model at step %d" % global_step)
                score = _evaluate(self._eval_fn, self._eval_input_fn,
                                  self._base_dir,
                                  self._session_config, self._save_path)
                tf.logging.info("%s at step %d: %f" %
                                (self._metric, global_step, score))

                _save_log(self._log_name, (self._metric, global_step, score))

                checkpoint_filename = os.path.join(self._base_dir,
                                                   "checkpoint")
                all_checkpoints = _read_checkpoint_def(checkpoint_filename)
                records = _read_score_record(self._record_name)
                latest_checkpoint = all_checkpoints[-1]
                record = [latest_checkpoint, score]
                added, removed, records = _add_to_record(records, record,
                                                         self._max_to_keep)

                if added is not None:
                    old_path = os.path.join(self._base_dir, added)
                    new_path = os.path.join(self._save_path, added)
                    old_files = tf.gfile.Glob(old_path + "*")
                    tf.logging.info("Copying %s to %s" % (old_path, new_path))

                    for o_file in old_files:
                        n_file = o_file.replace(old_path, new_path)
                        tf.gfile.Copy(o_file, n_file, overwrite=True)

                if removed is not None:
                    filename = os.path.join(self._save_path, removed)
                    tf.logging.info("Removing %s" % filename)
                    files = tf.gfile.Glob(filename + "*")

                    for name in files:
                        tf.gfile.Remove(name)

                _save_score_record(self._record_name, records)
                checkpoint_filename = checkpoint_filename.replace(
                    self._base_dir, self._save_path
                )
                _save_checkpoint_def(checkpoint_filename,
                                     [item[0] for item in records])

                best_score = records[0][1]
                tf.logging.info("Best score at step %d: %f" %
                                (global_step, best_score))

    def end(self, session):
        last_step = session.run(self._global_step)

        if last_step != self._timer.last_triggered_step():
            global_step = last_step
            tf.logging.info("Validating model at step %d" % global_step)
            score = _evaluate(self._eval_fn, self._eval_input_fn,
                              self._base_dir,
                              self._session_config)
            tf.logging.info("%s at step %d: %f" %
                            (self._metric, global_step, score))

            checkpoint_filename = os.path.join(self._base_dir,
                                               "checkpoint")
            all_checkpoints = _read_checkpoint_def(checkpoint_filename)
            records = _read_score_record(self._record_name)
            latest_checkpoint = all_checkpoints[-1]
            record = [latest_checkpoint, score]
            added, removed, records = _add_to_record(records, record,
                                                     self._max_to_keep)

            if added is not None:
                old_path = os.path.join(self._base_dir, added)
                new_path = os.path.join(self._save_path, added)
                old_files = tf.gfile.Glob(old_path + "*")
                tf.logging.info("Copying %s to %s" % (old_path, new_path))

                for o_file in old_files:
                    n_file = o_file.replace(old_path, new_path)
                    tf.gfile.Copy(o_file, n_file, overwrite=True)

            if removed is not None:
                filename = os.path.join(self._save_path, removed)
                tf.logging.info("Removing %s" % filename)
                files = tf.gfile.Glob(filename + "*")

                for name in files:
                    tf.gfile.Remove(name)

            _save_score_record(self._record_name, records)
            checkpoint_filename = checkpoint_filename.replace(
                self._base_dir, self._save_path
            )
            _save_checkpoint_def(checkpoint_filename,
                                 [item[0] for item in records])

            best_score = records[0][1]
            tf.logging.info("Best score: %f" % best_score)
