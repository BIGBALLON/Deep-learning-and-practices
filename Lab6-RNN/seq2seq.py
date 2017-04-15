import tensorflow as tf
import numpy as np 
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell

iterations = 15000
starter_learning_rate = 0.0008
output_matrix = False
seq_length = 30
train_length = 20
batch_size = 256
test_batch_size = 256
vocab_size = 257
embedding_dim = 100
memory_dim = 500

sess = tf.Session()

def train_batch(batch_size):
	X = [np.append(np.random.randint(1, vocab_size, size=train_length),np.zeros((seq_length-train_length,), dtype=np.int))
		 for _ in range(batch_size)]
	Y = X[:]
	# Dimshuffle to seq_len * batch_size
	X = np.array(X).T
	Y = np.array(Y).T
	# print(X.shape)

	W = np.full([batch_size, train_length], 1)
	W = np.append(W, np.zeros([batch_size, seq_length - train_length]), axis=1).T

	feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
	feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
	feed_dict.update({weights[t]: W[t] for t in range(seq_length)})

	_, loss_t, summary = sess.run([optimizer, loss, summary_op], feed_dict)
	return loss_t, summary

def cal_acc(X, dec_batch, num):
	X = X.T;
	Y = np.array(dec_batch).argmax(axis = 2).T
	correct = 0
	len_batch = len(Y)
	for i in range(len_batch):
		for j in range(num):
			if X[i][j] == Y[i][j]:
				correct += 1
	return correct/(len_batch*num)

def test(num,t,loss_t):
	X_batch = [np.append(np.random.randint(1, vocab_size, size=num),np.zeros((seq_length-num,), dtype=np.int))
				for _ in range(test_batch_size)]
	X_batch = np.array(X_batch).T

	W = np.full([batch_size, train_length], 1)
	W = np.append(W, np.zeros([batch_size, seq_length - train_length]), axis=1).T

	feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
	feed_dict.update({weights[t]: W[t] for t in range(seq_length)})
	dec_outputs_batch = sess.run(dec_outputs, feed_dict)
	acc = cal_acc( X_batch, dec_outputs_batch, num )
	print("test:%d/%d, itrations: %d, train_loss: %.5f, test_acc: %.5f%%." %( num ,seq_length, t, loss_t, acc*100.0) )
	if output_matrix:
		print("------------------------------matrix encode-----------------------------")
		print(X_batch)
		print("------------------------------matrix decode-----------------------------")
		for logits_t in dec_outputs_batch:
			print(logits_t.argmax(axis=1))
			

if __name__ == "__main__":

	enc_inp = [tf.placeholder(tf.int32, shape=(None,),name="inp%i" % t)  for t in range(seq_length)]
	dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="O")] + enc_inp[:-1])
	weights = [tf.placeholder(tf.float32, shape=(None,),name="weight%i" % t)  for t in range(seq_length)]
	labels = [tf.placeholder(tf.int32, shape=(None,),name="labels%i" % t) for t in range(seq_length)]
	# weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
	prev_mem = tf.zeros((batch_size, memory_dim))
	cell = MultiRNNCell([BasicLSTMCell(memory_dim)]*3)

	dec_outputs, dec_memory = legacy_seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)
	loss = legacy_seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
	optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(loss)

	tf.summary.scalar('loss', loss)
	summary_op = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('./logs', sess.graph)
	sess.run(tf.global_variables_initializer())

	for step in range(iterations):
		loss_t, summary = train_batch(batch_size)
		if step % 100 == 0 :
			print("itrations: %d, train_loss: %.5f." %( step, loss_t), end = '\r' )
		if step % 500 == 0 :
			summary_writer.add_summary(summary, step)
			summary_writer.flush()
			test(10,step,loss_t)
			test(20,step,loss_t)
			test(30,step,loss_t)
			# test(50,step,loss_t)
			print('-------------------------------------------------------------------------')
	