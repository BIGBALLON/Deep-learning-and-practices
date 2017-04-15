import tensorflow as tf
import numpy as np 
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell

tf.reset_default_graph()
sess = tf.Session()

iterations = 30000
starter_learning_rate = 0.0005
output_matrix = False
seq_length = 30
train_length = 20
batch_size = 128
test_batch_size = 256
vocab_size = 257
embedding_dim = 100
memory_dim = 500


enc_inp = [tf.placeholder(tf.int32, shape=(None,),name="inp%i" % t)  for t in range(seq_length)]
labels = [tf.placeholder(tf.int32, shape=(None,),name="labels%i" % t) for t in range(seq_length)]
weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")] + enc_inp[:-1])
prev_mem = tf.zeros((batch_size, memory_dim))
cell = BasicLSTMCell(memory_dim)

dec_outputs, dec_memory = legacy_seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size, embedding_dim)
loss = legacy_seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)


# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.97, staircase=True)
# momentum = 0.9
# optimizer = tf.train.MomentumOptimizer(starter_learning_rate,momentum, use_nesterov=True)
# train_op = optimizer.minimize(loss,global_step=global_step)

optimizer = tf.train.RMSPropOptimizer(starter_learning_rate,momentum=0.9)
train_op = optimizer.minimize(loss)


sess.run(tf.initialize_all_variables())


def train_batch(batch_size):
	X = [np.append(np.random.randint(1, vocab_size, size=train_length),np.zeros((seq_length-train_length,), dtype=np.int))
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

def cal_acc(X, dec_batch, num):
	X = X.T;
	Y = np.array(dec_batch).argmax(axis = 2).T
	if X.shape != Y.shape:
		return 0.0

	correct = 0
	len_batch = len(Y)
	for i in range(len_batch):
		cnt = 0
		for j in range(num):
			if X[i][j] == Y[i][j]:
				cnt = cnt + 1
		if cnt == num:
			correct += 1
	return correct/len_batch

def cal_acc2(X, dec_batch, num):
	X = X.T;
	Y = np.array(dec_batch).argmax(axis = 2).T
	if X.shape != Y.shape:
		return 0.0

	correct = 0
	len_batch = len(Y)
	for i in range(len_batch):
		for j in range(num):
			if X[i][j] == Y[i][j]:
				correct += 1
	return correct/(len_batch*num)

def test(num,t,loss_t):

	if num == seq_length:
		X_batch = [np.random.randint(1, vocab_size, size=num)
			   for _ in range(test_batch_size)]
	else:
		X_batch = [np.append(np.random.randint(1, vocab_size, size=num),np.zeros((seq_length-num,), dtype=np.int))
				for _ in range(test_batch_size)]
	
	X_batch = np.array(X_batch).T

	feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
	dec_outputs_batch = sess.run(dec_outputs, feed_dict)
	acc = cal_acc2( X_batch, dec_outputs_batch, num )
	print("seqlen: %d, itrations: %d, train_loss: %.5f, test_acc: %.5f%%." %( num, t, loss_t, acc*100.0) )
	if output_matrix:
		print("------------------------------matrix encode-----------------------------")
		print(X_batch)
		print("------------------------------matrix decode-----------------------------")
		for logits_t in dec_outputs_batch:
			print(logits_t.argmax(axis=1))
			

if __name__ == "__main__":
	for t in range(iterations):
		loss_t = train_batch(batch_size)
		if t % 1000 == 0 :
			test(10,t,loss_t)
			test(20,t,loss_t)
			test(30,t,loss_t)
	