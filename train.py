# coding: utf-8

import os
import math
import time
import json
import random

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from utils.data_iterator import BiTextIterator
from utils.data_utils import prepare_train_batch

from seq2seq import Seq2Seq


# config for data loading
tf.app.flags.DEFINE_string('source_vocabulary', 'data/sample.tok.en.json', 'Path to source vocabulary')
tf.app.flags.DEFINE_string('target_vocabulary', 'data/sample.tok.fr.json', 'Path to target vocabulary')
tf.app.flags.DEFINE_string('source_train_data', 'data/sample.tok.en', 'Path to source training data')
tf.app.flags.DEFINE_string('target_train_data', 'data/sample.tok.fr', 'Path to target training data')
tf.app.flags.DEFINE_string('source_valid_data', 'data/sample.tok.en', 'Path to target valid data')
tf.app.flags.DEFINE_string('target_valid_data', 'data/sample.tok.fr', 'Path to target valid data')

# config for networks
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_size', 50, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('layer_num', 1, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('en_vocab_size', 208, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('de_vocab_size', 216, 'Target vocabulary size')

tf.app.flags.DEFINE_boolean('use_att_decoding', False, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')

# config for training
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.app.flags.DEFINE_float('max_grad_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 4, 'Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 100, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 5, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_integer('max_seq_length', 34, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', 10, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 50, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 50, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'translate.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', False, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

FLAGS = tf.app.flags.FLAGS


def create_model(session, FLAGS):
    config = OrderedDict(sorted(FLAGS.__flags.items()))
    model = Seq2Seq(config, 'train')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("reloading model parameters...")
        model.restore(session, ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print("create new model parameters...")
        session.run(tf.global_variables_initializer())
    return model


def train():
    print("loading training data...")
    train_set = BiTextIterator(source=FLAGS.source_train_data,
                               target=FLAGS.target_train_data,
                               source_dict=FLAGS.source_vocabulary,
                               target_dict=FLAGS.target_vocabulary,
                               batch_size=FLAGS.batch_size,
                               maxlen=FLAGS.max_seq_length,
                               n_words_source=FLAGS.en_vocab_size,
                               n_words_target=FLAGS.de_vocab_size,
                               shuffle_each_epoch=FLAGS.shuffle_each_epoch,
                               sort_by_length=FLAGS.sort_by_length,
                               maxibatch_size=FLAGS.max_load_batches)
    if FLAGS.source_valid_data and FLAGS.target_valid_data:
        print ('Loading validation data..')
        valid_set = BiTextIterator(source=FLAGS.source_valid_data,
                                   target=FLAGS.target_valid_data,
                                   source_dict=FLAGS.source_vocabulary,
                                   target_dict=FLAGS.target_vocabulary,
                                   batch_size=FLAGS.batch_size,
                                   maxlen=None,
                                   n_words_source=FLAGS.en_vocab_size,
                                   n_words_target=FLAGS.de_vocab_size)
    else:
        valid_set = None

    # initialize TF session
    with tf.Session() as sess:
        # create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
        # create a new model or reload existing checkpoint
        model = create_model(sess, FLAGS)

        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # training loop
        print('training...')
        for epoch_idx in range(FLAGS.max_epochs):
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print('training is already complete. '
                      'current epoch: {}, max_epoch: {}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break

            for source_seq, target_seq in train_set:
                # print(source_seq)
                # print(target_seq)
                # get a batch from training data
                source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq,
                                                                             FLAGS.max_seq_length)
                # print(source, source_len, target, target_len)
                if source is None or target is None:
                    print('no samples under max_seq_length ', FLAGS.max_seq_length)
                    continue

                # execute a single training step
                step_loss, summary = model.train(sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                 decoder_inputs=target, decoder_inputs_length=target_len)

                loss += float(step_loss) / FLAGS.display_freq
                words_seen += float(np.sum(source_len + target_len))
                sents_seen += float(source.shape[0])  # batch_size

                if model.global_epoch_step.eval() % FLAGS.display_freq == 0:
                    avg_ppl = math.exp(float(loss)) if loss < 300 else float("inf")
                    time_elaspsed = time.time() - start_time
                    step_time = time_elaspsed / FLAGS.display_freq
                    words_per_sec = words_seen / time_elaspsed
                    sents_per_sec = sents_seen / time_elaspsed

                    print('Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(),
                          'ppl {0:.2f}'.format(avg_ppl), 'Step-time ', step_time,
                          '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec))

                    loss = 0
                    words_seen = 0
                    sents_seen = 0
                    start_time = time.time()

                    # record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # execute a validation step
                if valid_set and model.global_step.eval() % FLAGS.valid_freq == 0:
                    print('Validation step')
                    valid_loss = 0.0
                    valid_sents_seen = 0
                    for source_seq, target_seq in valid_set:
                        # Get a batch from validation parallel data
                        source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq)

                        # Compute validation loss: average per word cross entropy loss
                        step_loss, summary = model.eval(sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                        decoder_inputs=target, decoder_inputs_length=target_len)
                        batch_size = source.shape[0]

                        valid_loss += step_loss * batch_size
                        valid_sents_seen += batch_size
                        print('  {} samples seen'.format(valid_sents_seen))

                    valid_loss = valid_loss / valid_sents_seen
                    print('Valid perplexity: {0:.2f}'.format(math.exp(valid_loss)))

                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print("saving the model...")
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config, open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                              indent=2)

            # increase the epoch index of the model
            model.global_epoch_step_update.eval()
            print("Epoch {0:} DONE".format(model.global_epoch_step.eval()))

        print('saving the last model...')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                  indent=2)

    print("Training Terminated...")


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
