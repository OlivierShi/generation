# -*- coding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

import utils.data_utils as data_utils



class Seq2Seq(object):
    def __init__(self, config, mode):
        '''
        :param mode: train or decode. decode is for generating some example.
        '''
        assert mode.lower() in ['train', 'decode']
        self.config = config
        self.mode = mode.lower()

        self.cell_type = config['cell_type']
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['embedding_size']
        self.layer_num = config['layer_num']
        self.attention_type = config['attention_type']

        self.en_vocab_size = config['en_vocab_size']
        self.de_vocab_size = config['de_vocab_size']

        self.use_att_decoding = config['use_att_decoding']
        self.use_dropout = config['use_dropout']
        self.keep_prob = 1.0 - config['dropout_rate']

        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.max_grad_norm = config['max_grad_norm']
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_update = tf.assign(self.global_epoch_step, tf.add(self.global_epoch_step, 1))

        self.dtype = tf.float16 if config['use_fp16'] else tf.float32
        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')

        if self.mode == 'decode':
            self.max_decode_step = config['max_decode_step']

        self.build_model()

    def build_model(self):
        print("building model...")
        # build encoder and decoder networks
        # initialize placeholders and create feeding inputs
        self.init_placeholders()
        # build encoder
        self.build_encoder()
        # build decoder
        self.build_decoder()

        # merge all the training summaries
        self.summary_op = tf.summary.merge_all()




    def init_placeholders(self):
        print("creating placeholders...")
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='encoder_inputs')

        # encoder_inputs_length: [batch_size, ]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='encoder_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == 'train':
            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_inputs')
            # decoder_inputs: [batch_size, ]
            self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

            decoder_start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.start_token
            decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.end_token

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token, self.decoder_inputs], axis=1)
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1
            self.decoder_targets_train = tf.concat([self.decoder_inputs, decoder_end_token], axis=1)

    def build_encoder(self):
        print("building encoder...")
        with tf.variable_scope('encoder'):
            # build encoder_cell
            self.encoder_cell = self.build_encoder_cell()

            # initialize encoder_embeddings, Uniform(-sqrt(3), sqrt(3)), variance = 1
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)
            self.encoder_embeddings = tf.get_variable(name='embedding', shape=(self.en_vocab_size, self.embedding_size),
                                                      initializer=initializer, dtype=self.dtype)
            # embedded inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings,
                                                                  ids=self.encoder_inputs)
            input_layer = Dense(self.hidden_size, dtype=self.dtype, name='input_projection')
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # encode input sequences into context vectors:
            # encoder_ouputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size], if it has multi-layer,
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, dtype=self.dtype, time_major=False)

            tf.summary.histogram(name='encoder/'+self.encoder_embeddings.name, values=self.encoder_embeddings)

    def build_decoder(self):
        print("building decoder with attention...")
        with tf.variable_scope('decoder'):
            # building decoder_cell and decoder_initial_state
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

            # initialize decoder embeddings
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=self.dtype)

            self.decoder_embeddings = tf.get_variable(name='embedding', shape=(self.de_vocab_size, self.embedding_size),
                                                      initializer=initializer, dtype=self.dtype)

            input_layer = Dense(self.hidden_size, dtype=self.dtype, name='input_projection')
            output_layer = Dense(self.de_vocab_size, name='output_projection')

            if self.mode == 'train':
                print("training mode...")
                # decoder_inputs_embedded: [batch_size, max_time_step+1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(params=self.decoder_embeddings,
                                                                      ids=self.decoder_inputs_train)
                # go through input_layer
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)
                # helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                         sequence_length=self.decoder_inputs_length_train,
                                                         time_major=False,
                                                         name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                        helper=training_helper,
                                                        initial_state=self.decoder_initial_state,
                                                        output_layer=output_layer)
                # maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                # decoder_outputs_train: BasicDecoderOutput, (rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_outputs: [batch_size, max_time_step+1, de_vocab_size] if time_major=false
                #                                    [max_time_step+1, batch_size, de_vocab_size] if time_major=True
                (self.decoder_outputs_train, self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length
                ))

                # logit_train: [batch_size, max_time_step + 1, de_vocab_size]
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
                # predict using argmax
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')
                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')
                # computes per word average cross-entropy over a batch
                # use nn_ops.sparse_softmax_cross_entropy_with_logits by default
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                  targets=self.decoder_targets_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True)
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                tf.summary.histogram(name='decoder/' + self.decoder_embeddings.name, values=self.decoder_embeddings)

                # contruct graphs for minimizing loss
                self.init_optimizer()
            elif self.mode == 'decode':
                print("decoding mode by greedy decoder")
                # start_tokens: [batch_size,] 'int32' vector
                start_tokens = tf.ones(shape=(self.batch_size,), dtype=tf.int32) * data_utils.start_token

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))
                # feed inputs for greedy decoding: use the argamx of the output
                decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                end_token=data_utils.end_token,
                                                                embedding=embed_and_input_proj)
                inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                         helper=decoding_helper,
                                                         initial_state=self.decoder_initial_state,
                                                         output_layer=output_layer)
                # decoder_outputs_decode: BasicDecoderOutput, (rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, de_vocab_size]  if time_major=False
                #                                    [max_time_step, batch_size, de_vocab_size]  if time_major=True
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32  if output_time_major=False
                #                                   [max_time_step, batch_size], tf.int32  if output_time_major=True
                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    maximum_iterations=self.max_decode_step
                ))
                # decoder_pred_decode: [batch_size, max_time_step, 1]  (output_major_time=False)
                self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)


    def build_single_cell(self):
        cell_type = LSTMCell
        if self.cell_type.lower() == 'gru':
            cell_type = GRUCell
        cell = cell_type(self.hidden_size)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype, output_keep_prob=self.keep_prob_placeholder)
        return cell

    def build_encoder_cell(self):
        return MultiRNNCell([self.build_single_cell() for _ in range(self.layer_num)])

    def build_decoder_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length
        # building attention mechanism: default Bahdanau
        # 'Bahdanau': https://arxiv.org/abs/1409.0473
        self.attention_mechanism = attention_wrapper.BahdanauAttention(num_units=self.hidden_size,
                                                                       memory=encoder_outputs,
                                                                       memory_sequence_length=encoder_inputs_length)
        # 'Luong': https://arxiv.org/abs/1508.04025
        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = attention_wrapper.LuongAttention(
                num_units=self.hidden_size,
                memory=self.encoder_outputs,
                memory_sequence_length=self.encoder_inputs_length)

        # building decoder_cell
        self.decoder_cell_list = [self.build_single_cell() for _ in range(self.layer_num)]

        def att_decoder_input_fn(inputs, attention):
            if not self.use_att_decoding:
                return inputs

            _input_layer = Dense(self.hidden_size, dtype=self.dtype, name='att_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], axis=-1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # implement attention mechanism only on the top of decoder layer
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_size,
            cell_input_fn=att_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],  # last hidden state of last encode layer
            alignment_history=False,
            name='Attention_Wrapper'
        )
        initial_state = [state for state in encoder_last_state]
        initial_state[-1] = self.decoder_cell_list[-1].zero_state(batch_size=self.batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)
        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

    def init_optimizer(self):
        print("setting optimizer...")
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)
        # clip gradients
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        # update parameters
        self.updates = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def save(self, sess, path, var_list=None, global_step=None):
        # if var_list is None, return the list of all saveable variables
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print("model saved at %s..." % save_path)

    def restore(self, sess, path, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print("model restored from %s..." % path)

    def train(self, sess, encoder_inputs, encoder_inputs_length,
                          decoder_inputs, decoder_inputs_length):
        '''
        run a train step of model
        :param sess: tensorflow session.
        :param encoder_inputs: a numpy int matrix of [batch_size, max_source_time_step] as encoder inputs.
        :param encoder_inputs_length: a numpy int vector of [batch_size] as sequence lengths for each input in the batch
        :param decoder_inputs: [batch_size, max_target_time_step]
        :param decoder_inputs_length: [batch_size]
        :return: average loss, summary
        '''
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in the train mode!")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob


        output_feed = [self.updates,  # updates op for optimization
                       self.loss,   # average loss for current batch
                       self.summary_op]  # training summary

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]

    def eval(self, sess, encoder_inputs, encoder_inputs_length,
                         decoder_inputs, decoder_inputs_length):
        '''run an evaluation step of the model'''
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # input feeds for dropout
        input_feed[self.keep_prob_placeholder] = 1.0
        output_feed = [self.loss,  # loss for current batch
                       self.summary_op]  # evaluation summary

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        # generate target sequence given source sequence
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      decode=True)

        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.decoder_pred_decode]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]  # [batch_size, max_time_step, 1]

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                          decoder_inputs, decoder_inputs_length, decode):
        '''check and prepare input_feed'''
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                             "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                                 "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                                 "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed






