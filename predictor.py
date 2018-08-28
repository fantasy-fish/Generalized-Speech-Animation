import tensorflow as tf
from ops import *
from utils import *
import time

class Predictor(object):
    def __init__(self,sess,batch_size=100,checkpoint_dir="checkpoint",
                 n_hidden=3000,Kx=11,Ky=5,clean=True,first=False):
        self.sess = sess
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.n_hidden = n_hidden
        self.Kx = Kx
        self.Ky = Ky
        if clean:
            self.n_phonemes = 40
        else:
            self.n_phonemes = 67
        if first:
            self.n_shape = 16
        else:
            self.n_shape = 15
        self.build_model()


    def build_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size] + [self.Kx*self.n_phonemes],
									 name='phonemes')
        self.y_ = tf.placeholder(tf.float32, [self.batch_size] + [self.Ky*self.n_shape],
									 name='shape_parameters')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_probability')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        h0 = tf.tanh(dropout(linear(self.x,self.n_hidden,scope='h0'), self.keep_prob))
        h1 = tf.tanh(dropout(linear(h0, self.n_hidden, scope='h1'), self.keep_prob))
        h2 = tf.tanh(dropout(linear(h1, self.n_hidden, scope='h2'), self.keep_prob))
        self.y = linear(h2, self.Ky*self.n_shape, scope='output')
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y_,self.y))
        self.saver = tf.train.Saver()


    def train(self, config):
        x,y = load_data(config.data_dir,self.Kx,self.Ky,config.normalize,config.clean,config.first)
        #split the training&testing data
        n_train = int(len(x)*0.9)
        x_train,y_train = x[:n_train],y[:n_train]
        x_test,y_test = x[n_train:],y[n_train:]

        #summary ops
        train_summary_writer = tf.summary.FileWriter("./logs/train", self.sess.graph)
        test_summary_writer = tf.summary.FileWriter("./logs/test", self.sess.graph)
        summary_loss = tf.summary.scalar('training_loss', self.loss)
        train_summary_op = tf.summary.merge([summary_loss])
        test_summary_op = tf.summary.merge([summary_loss])
        #optimizer ops
        global_step = tf.Variable(0, name="tr_global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1) \
                    .minimize(self.loss,global_step=global_step)
        #initialize ops
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        learning_rate = config.learning_rate
        for epoch in range(config.epoch):
            batch_idxs = len(x_train)//config.batch_size
            for idx in range(batch_idxs):
                start_time = time.time()
                #training
                batch_x = x_train[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_y = y_train[idx * config.batch_size:(idx + 1) * config.batch_size]
                _,summary,step = self.sess.run([optimizer,train_summary_op,global_step],
                                               feed_dict={self.x:batch_x,
                                                          self.y_:batch_y,
                                                          self.keep_prob:config.keep_prob,
                                                          self.learning_rate:learning_rate})
                train_summary_writer.add_summary(summary, step)
                #testing
                _batch_idxs = len(x_test)//config.batch_size
                loss = 0
                for _idx in range(_batch_idxs):
                    _batch_x = x_test[_idx * config.batch_size:(_idx + 1) * config.batch_size]
                    _batch_y = y_test[_idx * config.batch_size:(_idx + 1) * config.batch_size]
                    _loss, _summary, _step = self.sess.run([self.loss, test_summary_op, global_step],
                                                     feed_dict={self.x: _batch_x,
                                                                self.y_: _batch_y,
                                                                self.keep_prob:1.0})
                    loss += _loss/_batch_idxs
                test_summary_writer.add_summary(_summary, _step)
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, loss))
                if (np.mod(step, 500)) == 2:
                    self.save(config.checkpoint_dir, step)
            if epoch % config.lr_decay_step == 0 and epoch != 0:
                learning_rate *= config.lr_decay_rate


    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "predictor"),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def predict(self,dir):
        x = load_test_data(dir,self.Kx)
        batch_idxs = len(x) // self.batch_size
        y=list()
        if batch_idxs>0:
            for idx in range(batch_idxs):
                batch_x = x[idx * self.batch_size:(idx + 1) * self.batch_size]
                pred = self.sess.run([self.y],feed_dict={self.x:batch_x,self.keep_prob:1.0})[0]
                pred = average_y(pred, self.Ky, self.n_shape)
                y += pred
            #predict for the remaining data
            last_idx = batch_idxs * self.batch_size
            num_remain = self.batch_size - (len(x)-last_idx)
            last_batch = x[last_idx:]+x[:num_remain]
            if len(last_batch) != self.batch_size:
                print("ASDASDASD")
            last_pred = self.sess.run([self.y],feed_dict={self.x:last_batch,self.keep_prob:1.0})[0]
            last_pred = average_y(last_pred, self.Ky, self.n_shape)  # convert 75 to 15
            y += last_pred[:-num_remain]
        else:
            num_remain = self.batch_size-len(x)
            batch_x = x + x[:num_remain]
            pred = self.sess.run([self.y], feed_dict={self.x: batch_x,self.keep_prob:1.0})[0]
            pred = pred[:len(x)]
            pred = average_y(pred,self.Ky,self.n_shape)#convert 75 to 15
            y = pred
        #write to the output file
        np.savetxt("prediction", y, fmt="%.8g")
        #add tanh in the last layer and try!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

