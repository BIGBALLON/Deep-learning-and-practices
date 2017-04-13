import tensorflow as tf
import numpy as np 
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell


tf.reset_default_graph()
sess = tf.Session()



seq_length = 20
batch_size = 64
vocab_size = 257
embedding_dim = 100
memory_dim = 500


enc_inp = [tf.placeholder(tf.int32, shape=(None,),
						  name="inp%i" % t)
		   for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
						name="labels%i" % t)
		  for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
		   for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
		   + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))

cell = BasicLSTMCell(memory_dim)

dec_outputs, dec_memory = legacy_seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)

loss = legacy_seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.5
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
										   1000, 0.975, staircase=True)
# Use simple momentum for the optimization.
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(starter_learning_rate,momentum, use_nesterov=True)
train_op = optimizer.minimize(loss,global_step=global_step)
sess.run(tf.initialize_all_variables())

def train_batch(batch_size):
	X = [np.random.randint(1, vocab_size, size = seq_length)
		 for _ in range(batch_size)]

	Y = X[:]
	
	# Dimshuffle to seq_len * batch_size
	X = np.array(X).T
	Y = np.array(Y).T
	# print(X.shape)
	feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
	feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

	_, loss_t = sess.run([train_op, loss], feed_dict)
	return loss_t

def cal_acc( X, dec_batch):
	X = X.T;
	Y = np.array(dec_batch).argmax(axis = 2).T
	if X.shape != Y.shape:
		return 0.0

	correct = 0
	len_seq = len(X[0])
	len_batch = len(X)
	for i in range(len_batch):
		if sum(X[i] == Y[i]) == len_seq:
			correct += 1
	return correct/len_batch


for t in range(30000):
	loss_t = train_batch(batch_size)
	# print(loss_t)
	if t % 1000 == 0 :
		X_batch = [np.random.randint(1, vocab_size, size=seq_length)
			   for _ in range(500)]
		X_batch = np.array(X_batch).T

		feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
		dec_outputs_batch = sess.run(dec_outputs, feed_dict)

		print("------------------------------------------------------------------")
		# print(X_batch)
		# for logits_t in dec_outputs_batch:
		# 	print(logits_t.argmax(axis=1))
		print("lr: %.5f,   itrations: %d,   train_loss: %.5f,   test_acc: %.5f%%." %(sess.run(learning_rate), t, loss_t, cal_acc( X_batch, dec_outputs_batch )*100.0) )