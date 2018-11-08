import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import re
from trainer import inference, align, loc_net, LPRVocab, encode, decode_beams
from general.utils import load_module


import glog as log
import os
import sys

import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Perform training of a detection model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()

def dataset_size(fname):
  count = 0
  with open(fname, 'r') as f:
    for _ in f:
      count += 1
  return count

lpr_patterns = [
  '^<[^>]*>[A-Z][0-9A-Z]{5}$',
  '^<[^>]*>[A-Z][0-9A-Z][0-9]{3}<police>$',
  '^<[^>]*>[A-Z][0-9A-Z]{4}<[^>]*>$',  # <Guangdong>, <Hebei>
  '^WJ<[^>]*>[0-9]{4}[0-9A-Z]$',
]


def lpr_pattern_check(label):
  for pattern in lpr_patterns:
    if re.match(pattern, label):
      return True
  return False


def read_data(height, width, channels_num, list_file_name, batch_size=10):
  reader = tf.TextLineReader()
  key, value = reader.read(list_file_name)
  filename, label = tf.decode_csv(value, [[''], ['']], ' ')

  image_filename = tf.read_file(filename)
  rgb_image = tf.image.decode_png(image_filename, channels=channels_num)
  rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
  resized_image = tf.image.resize_images(rgb_image_float, [height, width])
  resized_image.set_shape([height, width, channels_num])

  image_batch, label_batch, file_batch = tf.train.batch([resized_image, label, image_filename], batch_size=batch_size)
  return image_batch, label_batch, file_batch


def data_input(height, width, channels_num, filename, batch_size=10):
  files_string_producer = tf.train.string_input_producer([filename])
  image, label, filename = read_data(height, width, channels_num, files_string_producer, batch_size)

  image = align(image, loc_net(image))

  return image, label, filename


def find_best(predictions):
  for prediction in predictions:
    if lpr_pattern_check(prediction):
      return prediction
  return predictions[0]  # fallback


def edit_distance(s1, s2):
  m = len(s1) + 1
  n = len(s2) + 1
  tbl = {}
  for i in range(m): tbl[i, 0] = i
  for j in range(n): tbl[0, j] = j
  for i in range(1, m):
    for j in range(1, n):
      cost = 0 if s1[i - 1] == s2[j - 1] else 1
      tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)

  return tbl[i, j]


def accuracy(label, val, fname, vocab, r_vocab):
  pred = decode_beams(val, r_vocab)
  bs = len(label)
  acc, acc1 = 0, 0
  num = 0
  for i in range(bs):
    if not lpr_pattern_check(label[i].decode('utf-8')):  # GT label fails
      log.info('GT label fails: ' + label[i].decode('utf-8'))
      continue
    best = find_best(pred[i])
    # use classes lists instead of strings to get edd('<aaa>', '<bbb>') = 1
    edd = edit_distance(encode(label[i].decode('utf-8'), vocab), encode(best, vocab))
    if edd <= 1:
      acc1 += 1
    if label[i].decode('utf-8') == best:
      acc += 1
    else:
      if label[i] not in pred[i]:
        log.info('Check GT label: ' + label[i].decode('utf-8'))
      log.info(label[i].decode('utf-8') + ' -- ' + best + ' Edit Distance: ' + str(edd))
    num += 1
  return float(acc), float(acc1), num



def validate(config):
  if hasattr(config.eval, 'random_seed'):
    np.random.seed(config.eval.random_seed)
    tf.set_random_seed(config.eval.random_seed)
    random.seed(config.eval.random_seed)

  if hasattr(config.eval.execution, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.execution.CUDA_VISIBLE_DEVICES

  height, width, channels_num = config.input_shape
  max_lp_length = config.eval.max_lp_length
  beam_search_width = config.eval.beam_search_width # use > 1 for post-filtering over top-N
  rnn_cells_num = config.eval.rnn_cells_num


  vocab, r_vocab, num_classes = LPRVocab.create_vocab(config.train.train_list_file_path, config.eval.file_list_path)

  graph = tf.Graph()

  with graph.as_default():
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
      inp_data, label_val, file_names = data_input(height, width, channels_num,
                                                   config.eval.file_list_path, batch_size=1)

      prob = inference(rnn_cells_num, inp_data, num_classes)
      prob = tf.transpose(prob, (1, 0, 2))  # prepare for CTC

      data_length = tf.fill([tf.shape(prob)[1]], tf.shape(prob)[0])  # input seq length, batch size

      # result = tf.nn.ctc_greedy_decoder(prob, data_length, merge_repeated=True)
      result = tf.nn.ctc_beam_search_decoder(prob, data_length, merge_repeated=False, top_paths=beam_search_width)

      predictions = [tf.to_int32(p) for p in result[0]]
      d_predictions = tf.stack([tf.sparse_to_dense(p.indices, [1, max_lp_length], p.values, default_value=-1)
                                for p in predictions])

      init = tf.initialize_all_variables()
      saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

  # session
  conf = tf.ConfigProto()
  if hasattr(config.eval.execution, 'per_process_gpu_memory_fraction'):
    conf.gpu_options.per_process_gpu_memory_fraction = config.train.execution.per_process_gpu_memory_fraction
  if hasattr(config.eval.execution, 'allow_growth'):
    conf.gpu_options.allow_growth = config.train.execution.allow_growth

  sess = tf.Session(graph=graph, config=conf)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  sess.run(init)


  checkpoints_dir = os.path.join(config.model_dir, config.eval.model)
  latest_checkpoint = None
  wait_iters = 0

  if not os.path.exists(os.path.join(checkpoints_dir, 'eval')):
    os.mkdir(os.path.join(checkpoints_dir, 'eval'))
  writer = tf.summary.FileWriter(os.path.join(checkpoints_dir, 'eval'), sess.graph)


  while True:
    if config.eval.checkpoint != '':
      new_checkpoint = config.eval.checkpoint
    else:
      new_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    if latest_checkpoint != new_checkpoint:
      latest_checkpoint = new_checkpoint
      saver.restore(sess, latest_checkpoint)
      current_step = tf.train.load_variable(latest_checkpoint, 'global_step')

      test_size = dataset_size(config.eval.file_list_path)
      t = time.time()

      mean_accuracy, mean_accuracy_minus_1 = 0.0, 0.0

      steps = test_size
      num = 0
      for i in range(steps):
        val, slabel, fname = sess.run([d_predictions, label_val, file_names])
        a, a1, n = accuracy(slabel, val, fname, vocab, r_vocab)
        mean_accuracy += a
        mean_accuracy_minus_1 += a1
        num += n

      writer.add_summary(
        tf.Summary(value=[tf.Summary.Value(tag='evaluation/acc', simple_value=float(mean_accuracy / num)),
                          tf.Summary.Value(tag='evaluation/acc-1', simple_value=float(mean_accuracy_minus_1 / num))
                          ]), current_step)
      log.info('Test acc: {}'.format(mean_accuracy / num))
      log.info('Test acc-1: {}'.format(mean_accuracy_minus_1 / num))
      log.info('Time per step: {} for test size {}'.format(time.time() - t / steps, test_size))
    else:
      if wait_iters % 12 == 0:
        sys.stdout.write('\r')
        for _ in range(11 + wait_iters // 12):
          sys.stdout.write(' ')
        sys.stdout.write('\r')
        for _ in range(1 + wait_iters // 12):
          sys.stdout.write('|')
      else:
        sys.stdout.write('.')
      sys.stdout.flush()
      time.sleep(5)
      wait_iters += 1
    if config.eval.checkpoint != '':
      break


  coord.request_stop()
  coord.join(threads)
  sess.close()

def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  validate(cfg)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
