import numpy as np
import tensorflow as tf
sess = tf.Session()

# LSTM_CELL_Size = 4
#
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_Size, state_is_tuple=True)
# state = (tf.zeros([1, LSTM_CELL_Size]),)*2
# sample_input = tf.constant([[3,2,2,1,3,2]], dtype=tf.float32)
# print(sess.run(sample_input))
#
# with tf.variable_scope("LSTM_sample1"):
#     output, state_new = lstm_cell(sample_input, state)
# sess.run(tf.global_variables_initializer())
# print(sess.run(state_new))
#
# print(sess.run(output))

input_dim = 6
cells = []
LSTM_CELL_SIZE_1 = 4
cell1 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

LSTM_CELL_SIZE_2 = 5
cell2 = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)

sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input
output
sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})