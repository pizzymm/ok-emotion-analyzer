import tensorflow as tf
import time
import os
import data_helper
from cnn_model import TextCNN
from sklearn.cross_validation import train_test_split, KFold
import logging
import numpy as np

"""
 Run program like this way: python3 train_dev_test.py --cxt_type="NC"
"""

# Data path and parameter
tf.flags.DEFINE_string("positive_data_file", "./data/1.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("neutral_data_file", "./data/0.txt", "Data source for the neutral data.")
tf.flags.DEFINE_string("negtive_data_file", "./data/-1.txt", "Data source for the negative data.")
tf.flags.DEFINE_string("w2vModelPath", "/home/lvchao/vectors/vectors_300.bin", "the model of word2vec")
tf.flags.DEFINE_integer("sequence_length", 102, "the max of words in a sentence")

# Model Hyperparameters
tf.flags.DEFINE_string("cxt_type", "NC", "Adjacent feature strategyg (default: NC)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularizaion lambda (default: 1.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
network_input_width = 300
if FLAGS.cxt_type == "NC":
    network_input_width = FLAGS.embedding_dim
    logging.basicConfig(level=logging.CRITICAL, filename='C3-NC-CNN.log', filemode='w')
elif FLAGS.cxt_type == "LC":
    network_input_width = FLAGS.embedding_dim * 2
    logging.basicConfig(level=logging.CRITICAL, filename='C3-LC-CNN.log', filemode='w')
elif FLAGS.cxt_type == "RC":
    network_input_width = FLAGS.embedding_dim * 2
    logging.basicConfig(level=logging.CRITICAL, filename='C3-RC-CNN.log', filemode='w')
elif FLAGS.cxt_type == "LRC":
    network_input_width = FLAGS.embedding_dim * 3
    logging.basicConfig(level=logging.CRITICAL, filename='C3-LRC-CNN.log', filemode='w')


print("Parameters:")
logging.critical("Parameters:")
cnt = 0
for attr, value in sorted(FLAGS.__flags.items()):
    cnt += 1
    if cnt == len(FLAGS.__flags.items()):
        print("{}={}\n".format(attr.upper(), value))
        logging.critical("{}={}\n".format(attr.upper(), value))
    else:
        print("{}={}".format(attr.upper(), value))
        logging.critical("{}={}".format(attr.upper(), value))


# Load data
print("Loading data...")
logging.critical("Loading data...")
x_data, y_data = data_helper.load_data_label(FLAGS.positive_data_file, FLAGS.negtive_data_file, FLAGS.neutral_data_file, FLAGS.w2vModelPath,
                                             FLAGS.sequence_length, FLAGS.cxt_type)
print("Data loaded!\n")
logging.critical("Data loaded!\n".format())


# Split train/Dev/Test(10-fold-validation)
CV, final_accuracy = 0, 0
SKF = KFold(len(y_data), n_folds=10, shuffle=True, random_state=0)
for train_indices, test_indices in SKF:
    x_train_pre = x_data[train_indices]
    y_train_pre = y_data[train_indices]
    x_test = x_data[test_indices]
    y_test = y_data[test_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_train_pre, y_train_pre, test_size=0.1, random_state=0)

    # Training
    print("CV {} Training...".format(CV))
    logging.critical("CV {} Training...".format(CV))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(sequence_length=FLAGS.sequence_length,
                          num_classes=y_train.shape[1],
                          embedding_size=network_input_width,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                          num_filters=FLAGS.num_filters,
                          l2_reg_lambda=FLAGS.l2_reg_lambda)

            # define the Training produce
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # output directory for models and summaries
            timestamp = str(FLAGS.cxt_type) + str("_") + str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

            # Summary for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "train_summary", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "dev_summary", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            """
            Train the model on the train data
            """
            def train_step(x_batch, y_batch):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)
                return step, loss, accuracy

            """
            Evaluate the model on the dev/test data
            """
            def dev_test_step(x_batch, y_batch, writer=None):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy, num_correct, predictions = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.num_correct, cnn.predictions], feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return num_correct, predictions

            # Train
            for num_epoch in range(FLAGS.num_epochs):
                train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, shuffle=True)
                start_time = time.time()
                losses, accuracys, cnt = 0, 0, 0
                for train_batch in train_batches:
                    x_batch, y_batch = zip(*train_batch)
                    step, loss, accuracy = train_step(x_batch, y_batch)
                    last_train_step = step
                    last_train_loss = loss
                    last_train_accuracy = accuracy
                    losses += loss
                    accuracys += accuracy
                    cnt += 1
                train_loss = losses / cnt
                train_accuracy = accuracys / cnt

                # Dev
                dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, shuffle=False)
                total_dev_correct = 0
                for dev_batch in dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    num_dev_correct, _ = dev_test_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                    total_dev_correct += num_dev_correct
                dev_accuracy = float(total_dev_correct) / len(y_dev)
                print("Epoch: {}, Train_time: {:.2f}s, Train_accuracy: {:.4f}, Train_loss: {:.4f}, Last_step:{}, Last_train_loss: {:.4f}, Last_train_accuracy: {:.4f}, Dev_accuracy: {:.4f}".format(num_epoch, time.time() - start_time, train_accuracy, train_loss, last_train_step, last_train_loss, last_train_accuracy, dev_accuracy))
                logging.critical("Epoch: {}, Train_time: {:.2f}s, Train_accuracy: {:.4f}, Train_loss: {:.4f}, Last_step:{}, Last_train_loss: {:.4f}, Last_train_accuracy: {:.4f}, Dev_accuracy: {:.4f}".format(num_epoch, time.time() - start_time, train_accuracy, train_loss, last_train_step, last_train_loss, last_train_accuracy, dev_accuracy))

            # Test
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), FLAGS.batch_size, shuffle=False)
            all_lables, all_predictions, total_test_correct = [], [], 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct, batch_predictions = dev_test_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct

                for y_test_item in y_test_batch:
                    if y_test_item[0] == 1:
                        all_lables = np.concatenate((all_lables, [0]))
                    elif y_test_item[1] == 1:
                        all_lables = np.concatenate((all_lables, [1]))
                    elif y_test_item[2] == 1:
                        all_lables = np.concatenate((all_lables, [2]))
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            # Compute Precision, Recall, F1-measure, Accuracy
            num_neg, num_neu, num_pos = 0, 0, 0
            num_neg_pos, num_neg_neu, num_neg_neg = 0, 0, 0
            num_neu_pos, num_neu_neu, num_neu_neg = 0, 0, 0
            num_pos_pos, num_pos_neu, num_pos_neg = 0, 0, 0
            for i in range(len(all_lables)):
                if all_lables[i] == 0:
                    num_neg += 1
                    if all_predictions[i] == 0:
                        num_neg_neg += 1
                    elif all_predictions[i] == 1:
                        num_neg_neu += 1
                    elif all_predictions[i] == 2:
                        num_neg_pos += 1
                elif all_lables[i] == 1:
                    num_neu += 1
                    if all_predictions[i] == 0:
                        num_neu_neg += 1
                    elif all_predictions[i] == 1:
                        num_neu_neu += 1
                    elif all_predictions[i] == 2:
                        num_neu_pos += 1
                elif all_lables[i] == 2:
                    num_pos += 1
                    if all_predictions[i] == 0:
                        num_pos_neg += 1
                    elif all_predictions[i] == 1:
                        num_pos_neu += 1
                    elif all_predictions[i] == 2:
                        num_pos_pos += 1

            if (num_pos_neg + num_neu_neg + num_neg_neg) != 0:
                P_neg = num_neg_neg / (num_pos_neg + num_neu_neg + num_neg_neg)
            else:
                P_neg = 0.0
            if num_neg != 0:
                R_neg = num_neg_neg / num_neg
            else:
                R_neg = 0.0
            if (P_neg + R_neg) != 0:
                F_neg = 2 * P_neg * R_neg / (P_neg + R_neg)
            else:
                F_neg = 0.0

            if (num_pos_neu + num_neu_neu + num_neg_neu) != 0:
                P_neu = num_neu_neu / (num_pos_neu + num_neu_neu + num_neg_neu)
            else:
                P_neu = 0.0
            if num_neu != 0:
                R_neu = num_neu_neu / num_neu
            else:
                R_neu = 0.0
            if (P_neu + R_neu) != 0:
                F_neu = 2 * P_neu * R_neu / (P_neu + R_neu)
            else:
                F_neu = 0.0

            if (num_pos_pos + num_neu_pos + num_neg_pos) != 0:
                P_pos = num_pos_pos / (num_pos_pos + num_neu_pos + num_neg_pos)
            else:
                P_pos = 0.0
            if num_pos != 0:
                R_pos = num_pos_pos / num_pos
            else:
                R_pos = 0.0
            if (P_pos + R_pos) != 0:
                F_pos = 2 * P_pos * R_pos / (P_pos + R_pos)
            else:
                F_pos = 0.0

            test_accuracy = float(total_test_correct) / len(y_test)
            final_accuracy += test_accuracy

            print("CV {}: P_neg: {:.4f}, R_neg: {:.4f}, F_neg: {:.4f}, P_neu: {:.4f}, R_neu: {:.4f}, F_neu: {:.4f}, P_pos: {:.4f}, R_pos: {:.4f}, F_pos: {:.4f}, Test Accuracy: {:.4f}\n".format(CV, P_neg, R_neg, F_neg, P_neu, R_neu, F_neu, P_pos, R_pos, F_pos, test_accuracy))
            logging.critical("CV {}: P_neg: {:.4f}, R_neg: {:.4f}, F_neg: {:.4f}, P_neu: {:.4f}, R_neu: {:.4f}, F_neu: {:.4f}, P_pos: {:.4f}, R_pos: {:.4f}, F_pos: {:.4f}, Test Accuracy: {:.4f}\n".format(CV, P_neg, R_neg, F_neg, P_neu, R_neu, F_neu, P_pos, R_pos, F_pos, test_accuracy))
            CV += 1

print("Average Test Accuracy: {:.4f}".format(final_accuracy / float(10)))
logging.critical("Average Test Accuracy: {:.4f}".format(final_accuracy / float(10)))
