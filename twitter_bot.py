import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.append('/u/tyang21/Documents/libs')
import tensorflow as tf
import random


def encode_vocab(text, vocab):
    return [vocab.index(x)+1 for x in text if x in vocab]


def decode_vocab(array, vocab):
    return ''.join([vocab[x-1] for x in array])


def get_data(file_path, vocab, window, overlap):
    lines = [_.strip() for _ in open(file_path).readlines()]

    while True:
        random.shuffle(lines)
        for line in lines:
            text = encode_vocab(line, vocab)
            for start in range(0, len(text)-window, int(overlap)):
                chunk = text[start: start+window]
                chunk += [0]*(window-len(chunk))
                yield chunk


def get_batch(stream, batch_size,):
    batch = []
    for ele in stream:
        batch.append(ele)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


class Char_RNN(object):
    def __init__(self):
        self.window = 50
        self.vocab = ("$%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    " '\"_abcdefghijklmnopqrstuvwxyz{|}@#➡📈")
        self.hidden_size = [128, 256]
        self.batch = 64
        self.lr = 0.003
        self.seq = tf.placeholder(dtype=tf.int32, shape=[None, None])  # place holder for input batch of window size sequences
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.num_steps = 50  # window size
        self.path = './trump_tweets.txt'
        self.skip_step = 1
        self.gen_length = 200
        self.temp = tf.constant(1.5)

    def create_rnn(self, seq):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_size]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        batch = tf.shape(seq)[0]
        zero_state = cells.zero_state(batch, tf.float32)  # first param : batch size
        self.in_state = tuple(tf.placeholder_with_default(state, [None, state.shape[1]]) for state in zero_state)  # initial state
        # compute the real length of each seq?
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq),2),1)
        self.out, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state)

    def build_rnn(self):
        seq = tf.one_hot(self.seq, len(self.vocab))
        self.create_rnn(seq)
        self.logits = tf.layers.dense(self.out, len(self.vocab), None)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1], labels=seq[:, 1:])
        self.loss = tf.reduce_sum(loss)
        self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0]
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step= self.gstep)

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration = self.gstep.eval()
            stream = get_data(self.path, self.vocab, self.num_steps, self.num_steps/2)
            data = get_batch(stream, self.batch)
            while True:
                batch = next(data)
                batch_loss, _ = sess.run([self.loss, self.opt],{self.seq:batch})
                if iteration % self.skip_step == 0:
                    print('Iteration {}.  Loss {}.'.format(iteration, batch_loss))
                    self.inference(sess)
                iteration += 1

    def inference(self, sess):
        seed = ['W','I','R','@']
        for sd in seed:
            sentence = sd  # initial sentence
            state = None
            for _ in range(self.gen_length):
                batch = [encode_vocab(sentence[-1], self.vocab)]  # why sentence[-1]?
                feed = {self.seq: batch}
                if state is not None:
                    for i in range(len(state)):
                        feed.update({self.in_state[i]: state[i]})
                index, state = sess.run([self.sample, self.out_state], feed)
                character = decode_vocab(index, self.vocab)
                sentence += character
            print('\t' + sentence)


if __name__ == '__main__':
    model = Char_RNN()
    model.build_rnn()
    model.train()