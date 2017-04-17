import tensorflow as tf
import numpy as np 
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, GRUCell

# global variable
iterations = 13000 + 1
starter_learning_rate = 0.001
output_matrix = False

###############################
seq_length = 30
train_length = 20
test_len_list = [10,20,30,50]
test_index = [0,1,2]
###############################

batch_size = 128
word_size = 256 + 2
embedding_dim = 100
memory_dim = 500

logs_path = './logs'

# define placeholder
enc_inp = [tf.placeholder(tf.int32, shape=(None,),name="inp%i" % t)  for t in range(seq_length)]
dec_inp = [tf.placeholder(tf.int32, shape=(None,),name="dec%i" % t)  for t in range(seq_length)]
weights = [tf.placeholder(tf.float32, shape=(None,),name="weight%i" % t)  for t in range(seq_length)]

# calculate the accuracy of testing
def cal_acc(X, dec_batch, num):
	Y = np.array(dec_batch).argmax(axis = 2)
	correct = 0
	len_batch = len(Y)
	for i in range(len_batch):
		for j in range(num):
			if X[i][j] == Y[i][j]:
				correct += 1
	return correct/(len_batch*num)

# set feed dictionary
def set_feed_dict(used_len):
	X = [np.append(np.random.randint(1, 257, size=used_len),np.zeros((seq_length-used_len,), dtype=np.int)) for _ in range(batch_size)]
	D = np.full([batch_size, seq_length], 257)
	W = np.full([batch_size, seq_length], 1)

	# W = np.full([batch_size, used_len], 1)
	# W = np.append(W, np.zeros([batch_size, seq_length - used_len]), axis=1)

	feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
	feed_dict.update({dec_inp[t]: D[t] for t in range(seq_length)})
	feed_dict.update({weights[t]: W[t] for t in range(seq_length)})
	return feed_dict, X

# test all kinds of optimizer
def set_optimizer(lr):
	# return tf.train.AdamOptimizer(lr)
	# return tf.train.GradientDescentOptimizer(lr)
	# return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
	return tf.train.RMSPropOptimizer(learning_rate=0.0003, momentum=0.9)

# set cell of RNN
def set_cell():
	return BasicLSTMCell(memory_dim)
	# return GRUCell(memory_dim)
	# return MultiRNNCell( [ BasicLSTMCell(memory_dim)] * 3 )
	# return MultiRNNCell( [ GRUCell(memory_dim)] * 3 )


# training and testing
def train(batch_size):

	# using embedding_rnn_seq2seq model
	dec_outputs, _ = legacy_seq2seq.embedding_rnn_seq2seq(
			encoder_inputs = enc_inp, 
			decoder_inputs = dec_inp, 
			cell = set_cell(), 
			num_encoder_symbols = word_size, 
			num_decoder_symbols = word_size, 
			embedding_size = embedding_dim)

	# calc loss
	loss = legacy_seq2seq.sequence_loss(dec_outputs, enc_inp, weights)

	# using optimizer
	optimizer = set_optimizer(starter_learning_rate).minimize(loss)

	# define session and summary operation
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	tf.summary.scalar('loss', loss)
	summary_op = tf.summary.merge_all()

	train_writer = tf.summary.FileWriter(logs_path+r'/train', sess.graph)
	test_writer = []
	test_writer.append(tf.summary.FileWriter(logs_path+r'/test_10'))
	test_writer.append(tf.summary.FileWriter(logs_path+r'/test_20'))
	test_writer.append(tf.summary.FileWriter(logs_path+r'/test_30'))
	test_writer.append(tf.summary.FileWriter(logs_path+r'/test_50'))

	# training 
	for step in range(iterations):

		feed_dict, _ = set_feed_dict(train_length)
		_, train_loss, summary = sess.run([optimizer, loss, summary_op], feed_dict)
		train_writer.add_summary(summary, step)

		if step % 10 == 0 :
			print("itrations: %d, train_loss: %.5f." %( step, train_loss ), end = '\r' )

		# testing
		if step % 500 == 0 :
			for i in test_index:
				# test length
				test_len = test_len_list[i]
				feed_dict, X_batch = set_feed_dict(test_len)
				dec_outputs_batch, test_loss, summary = sess.run([dec_outputs,loss,summary_op], feed_dict)	
				test_writer[i].add_summary(summary, step)
				test_writer[i].flush()

				testing_acc = cal_acc(X_batch, dec_outputs_batch, test_len)

				print("test:%d/%d, itrations: %d, test_loss: %.5f, test_acc: %.5f%%." %( test_len ,seq_length, step, test_loss, testing_acc*100.0) )
				if output_matrix:
					print("----------------------------------matrix encode---------------------------------")
					for i in range(4):
						print(X_batch[i])
					print("----------------------------------matrix decode---------------------------------")
					Y_batch = np.array(dec_outputs_batch).argmax(axis = 2)
					for i in range(4):
						print(Y_batch[i])
				print('--------------------------------------------------------------------------------')


if __name__ == "__main__":
	train(batch_size)
		
