import tensorflow as tf
from predictor import Predictor
import os
import shutil

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train [20]")
flags.DEFINE_float("learning_rate", 1e-5, "Learning rate for adam")
flags.DEFINE_integer("lr_decay_step",2, "Number of steps when the learning rate is decayed eachtime")
flags.DEFINE_float("lr_decay_rate",0.5, "Rate at which the learning rate decays")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_float("lr_decay_step", 100, "number of epochs when learning rate is decayed by 1/2")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "../../data/KB-2k/", "Directory name to load the training data [data_dir]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("clean", True, "True if clean the phonemes")
flags.DEFINE_integer("n_hidden",3000,"Number of dimensions for y")
flags.DEFINE_boolean("first",False,"whether to keep the first dimension")
flags.DEFINE_boolean("normalize",True,"whether to keep the first dimension")
flags.DEFINE_float("keep_prob",0.5,"drop out rate")
flags.DEFINE_integer("Kx",11,"input window width")
flags.DEFINE_integer("Ky",5,"output window width")
flags.DEFINE_string("test_data", "test/si635/si635.TextGrid", "Directory name to load the testing data [data_dir]")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.is_train:
        if os.path.exists(FLAGS.checkpoint_dir):
            shutil.rmtree(FLAGS.checkpoint_dir)
        if os.path.exists('logs'):
            shutil.rmtree('logs')
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        predictor = Predictor(sess,
                              batch_size=FLAGS.batch_size,
                              checkpoint_dir=FLAGS.checkpoint_dir,
                              n_hidden=FLAGS.n_hidden,
                              Kx=FLAGS.Kx,
                              Ky=FLAGS.Ky,
                              clean=FLAGS.clean,
                              first=FLAGS.first)
        if FLAGS.is_train:
            predictor.train(FLAGS)
        else:
            if predictor.load(FLAGS.checkpoint_dir):
                predictor.predict(FLAGS.test_data)

if __name__ == '__main__':
    tf.app.run()