import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate
from core.utils import load_coco_data


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer   

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):

        ######################################################
        # move to each epoch to solve huge data load problem #
        ######################################################

        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        # features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        caption_idxs = {}
        for i in range(len(image_idxs)):
            if image_idxs[i] not in caption_idxs:
                caption_idxs[image_idxs[i]] = [i]
            else:
                caption_idxs[image_idxs[i]].append(i)

        # val_features = self.val_data['features']
        val_features = load_val_data()
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))


        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            _, _, generated_captions = self.model.build_sampler(max_len=20)

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
           
        # summary op   
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     tf.summary.histogram(var.op.name+'/gradient', grad)
        
        summary_op = tf.summary.merge_all() 

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            print "The number of epoch: %d" %self.n_epochs
            print "Batch size: %d" %self.batch_size

            for e in range(self.n_epochs):
                # load data 9 times to solve huge data load problem #
                cur_iteration = 0
                for data_cnt in range(9):
                    print "----------------------------------------------------"
                    print "Loading data (part %d / 9) " %(int(data_cnt)+1)

                    features = hickle.load(os.path.join('./data/train', 'train.features%d.hkl' % data_cnt))
                    
                    total_num = features.shape[0]
                    print "Load success (data size: %d) " %total_num
                    print "Iterations: %d" %n_iters_per_epoch
                    print "----------------------------------------------------"

                    index_st = data_cnt * 10000
                    index_ed = index_st + total_num
                    part_features = features
                    part_captions = []
                    part_image_idxs = []

                    for idx in range(total_num):
                        for caption_idx in caption_idxs[index_st + idx]:
                            part_captions.append(captions[caption_idx])
                            part_image_idxs.append(idx)
                    part_captions = np.asarray(part_captions)
                    part_image_idxs = np.asarray(part_image_idxs)
                    part_iters = int(np.ceil(float(part_captions.shape[0])/self.batch_size))
                    

                    rand_idxs = np.random.permutation(part_captions.shape[0])
                    part_captions = part_captions[rand_idxs]
                    part_image_idxs = part_image_idxs[rand_idxs]

                    for i in range(part_iters):
                        captions_batch = part_captions[i*self.batch_size:(i+1)*self.batch_size]
                        part_image_idxs_batch = part_image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                        features_batch = part_features[part_image_idxs_batch]
                        feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                        _, l = sess.run([train_op, loss], feed_dict)
                        curr_loss += l

                        # write summary for tensorboard visualization
                        if cur_iteration % 10 == 0:
                            summary = sess.run(summary_op, feed_dict)
                            summary_writer.add_summary(summary, e*n_iters_per_epoch + cur_iteration)

                        if (cur_iteration+1) % self.print_every == 0:
                            print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, cur_iteration+1, l)
                            ground_truths = part_captions[part_image_idxs == part_image_idxs_batch[0]]
                            decoded = decode_captions(ground_truths, self.model.idx_to_word)
                            for j, gt in enumerate(decoded):
                                print "Ground truth %d: %s" %(j+1, gt)                    
                            gen_caps = sess.run(generated_captions, feed_dict)
                            decoded = decode_captions(gen_caps, self.model.idx_to_word)
                            print "Generated caption: %s\n" %decoded[0]
                        cur_iteration = cur_iteration + 1

                    print "Current( epoch %d / part %d )  loss: %f" %(e+1, data_cnt+1, curr_loss)

                print "----------------------------------------------------"
                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                print "----------------------------------------------------"
                prev_loss = curr_loss
                curr_loss = 0
                
                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    for i in range(n_iters_val):
                        features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        feed_dict = {self.model.features: features_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)  
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                    
                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)
            
         
    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" %decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights 
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(14,14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    # plt.show()
                    plt.savefig('./jpg/soft_%d.jpg'%(n))
                    

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)  
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))