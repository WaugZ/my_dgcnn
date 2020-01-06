import tensorflow as tf
import os
import sys
import importlib
import time
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_CLIP = 0.01

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


class BigModel:
    def __init__(self, args, model_type):
        self.learning_rate = args.learning_rate
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.num_points = 1024
        self.num_classes = 40
        self.checkpoint_file = "bigmodel"
        self.temperature = args.temperature
        self.checkpoint_path = args.checkpoint_path
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type
        self.MODEL = importlib.import_module(args.model)  # import network module
        self.quantize_delay = args.quantize_delay_t
        self.DYNAMIC = True if args.dynamic_t < 0 else False
        self.STN = True if args.stn_t < 0 else False
        self.SCALE = args.scale_t
        self.CONCAT = True if args.concat_t == 1 else False
        self.BN_DECAY_DECAY_STEP = float(args.decay_step)

        self.build_model(args.model_type)
        self.saver = tf.train.Saver()

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch * self.batch_size,
            self.BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_momentum = tf.maximum(BN_DECAY_CLIP, bn_momentum)
        return bn_momentum
        # bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        # return bn_decay

    def train_one_epoch(self, sess, train_writer):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        # Shuffle train files
        train_file_idxs = np.arange(0, len(TRAIN_FILES))
        np.random.shuffle(train_file_idxs)

        mean_acc = 0
        for fn in range(len(TRAIN_FILES)):
            print('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
            current_data = current_data[:, 0:self.num_points, :]
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // self.batch_size

            total_correct = 0
            total_seen = 0
            loss_sum = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                # Augment batched point clouds by rotation and jittering
                rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
                jittered_data = provider.jitter_point_cloud(rotated_data)
                jittered_data = provider.random_scale_point_cloud(jittered_data)
                jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
                jittered_data = provider.shift_point_cloud(jittered_data)

                if not self.quantize_delay:
                    feed_dict = {
                        # ops['pointclouds_pl']: jittered_data,
                        self.pointclouds_pl: current_data[start_idx:end_idx, :, :],
                        self.labels_pl: current_label[start_idx:end_idx],
                        self.is_training: is_training,
                    }
                else:
                    feed_dict = {
                        # ops['pointclouds_pl']: jittered_data,
                        self.pointclouds_pl: current_data[start_idx:end_idx, :, :],
                        self.labels_pl: current_label[start_idx:end_idx],
                        # ops['is_training_pl']: is_training,
                    }
                summary, step, _, loss_val, pred_val = sess.run([self.merged_summary_op, self.batch, self.train_op,
                                                              self.total_loss, self.prediction],
                                                                feed_dict=feed_dict)
                # sess.run(ops['mask_update_op'])
                train_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum(pred_val == current_label[start_idx:end_idx])
                total_correct += correct
                total_seen += self.batch_size
                loss_sum += loss_val

            print('mean loss: %f' % (loss_sum / float(num_batches)))
            print('accuracy: %f' % (total_correct / float(total_seen)))

            mean_acc += total_correct / float(total_seen)
        return mean_acc / len(TRAIN_FILES)

    def eval_one_epoch(self, sess):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self.num_classes)]
        total_correct_class = [0 for _ in range(self.num_classes)]

        for fn in range(len(TEST_FILES)):
            print('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
            current_data = current_data[:, 0:self.num_points, :]
            # current_data = np.expand_dims(current_data, axis=-2)
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // self.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                feed_dict = {self.pointclouds_pl: current_data[start_idx:end_idx, :, :],
                             self.labels_pl: current_label[start_idx:end_idx],
                             self.is_training: is_training
                             }

                summary, step, loss_val, pred_val = sess.run([self.merged_summary_op, self.batch,
                                                              self.total_loss, self.prediction], feed_dict=feed_dict)
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum(pred_val == current_label[start_idx:end_idx])
                total_correct += correct
                total_seen += self.batch_size
                loss_sum += (loss_val * self.batch_size)
                for i in range(start_idx, end_idx):
                    l = current_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i - start_idx] == l)

        print('eval mean loss: %f' % (loss_sum / float(total_seen)))
        acc = total_correct / float(total_seen)
        print('eval accuracy: %f' % acc)
        aca = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        print('eval avg class acc: %f' % aca)

        return acc

    # Create model
    def build_model(self, model_type):
        self.pointclouds_pl = self.MODEL.placeholder_input(self.batch_size, self.num_points)
        self.labels_pl = self.MODEL.placeholder_label(self.batch_size)
        if not self.quantize_delay:
            self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        else:
            self.is_training = True
            if model_type == "student":
                self.is_training = False

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        self.batch = tf.Variable(0)
        # bn_decay = BN_INIT_DECAY
        bn_decay = self.get_bn_decay(self.batch)
        tf.summary.scalar('bn_decay', bn_decay)

        # Get model
        with tf.name_scope("%s" % (self.model_type)), tf.variable_scope("%s" % (self.model_type)):
            logits, end_points = self.MODEL.get_network(self.pointclouds_pl, self.is_training,
                                                 bn_decay=bn_decay,
                                                 dynamic=self.DYNAMIC,
                                                 STN=self.STN,
                                                 scale=self.SCALE,
                                                 concat_fea=self.CONCAT)

        with tf.name_scope("%sprediction" % (self.model_type)), tf.variable_scope("%sprediction" % (self.model_type)):
            self.prediction = tf.nn.softmax(logits)
            # Evaluate model
            correct = tf.equal(tf.argmax(logits, 1), tf.cast(self.labels_pl, tf.int64))
            self.accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self.batch_size)

        if model_type != "student":
            with tf.name_scope("%soptimization" % (self.model_type)), tf.variable_scope(
                            "%soptimization" % (self.model_type)):
                # Define loss and optimizer
                loss = self.MODEL.get_loss(logits, self.labels_pl, end_points)
                regularization_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.model_type)
                all_losses = []
                all_losses.append(loss)
                all_losses.append(tf.add_n(regularization_losses))
                self.total_loss = tf.add_n(all_losses)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_type)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                with tf.control_dependencies([tf.group(*update_ops)]):
                    self.train_op = optimizer.minimize(self.total_loss, global_step=self.batch)

                # tf.summary.scalar('loss', loss)
                tf.summary.scalar('loss', self.total_loss)

            with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
                            "%ssummarization" % (self.model_type)):
                tf.summary.scalar("loss", self.total_loss)
                # Create a summary to monitor accuracy tensor
                tf.summary.scalar("accuracy", self.accuracy)

                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)

                # Merge all summaries into a single op

                # If using TF 1.6 or above, simply use the following merge_all function
                # which supports scoping
                # self.merged_summary_op = tf.summary.merge_all(scope=self.model_type)

                # Explicitly using scoping for TF versions below 1.6

                def mymergingfunction(scope_str):
                    with tf.name_scope("%s_%s" % (self.model_type, "summarymerger")), tf.variable_scope(
                                    "%s_%s" % (self.model_type, "summarymerger")):
                        from tensorflow.python.framework import ops as _ops
                        key = _ops.GraphKeys.SUMMARIES
                        summary_ops = _ops.get_collection(key, scope=scope_str)
                        if not summary_ops:
                            return None
                        else:
                            return tf.summary.merge(summary_ops)

                self.merged_summary_op = mymergingfunction(self.model_type)

    def start_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)

    def close_session(self):
        self.sess.close()

    def train(self):

        # Initialize the variables (i.e. assign their default value)
        self.sess.run(tf.global_variables_initializer())

        print("Starting Training")

        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        max_accuracy = 0
        if self.checkpoint_path:
            print("restore model parameters from {}".format(self.checkpoint_path))
            scope_saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_type))
            scope_saver.restore(self.sess, self.checkpoint_path)
        for epoch in range(self.num_epoch + 1):
            print(('**** EPOCH %03d ****' % (epoch))
                       + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '****')
            ma = self.train_one_epoch(self.sess, train_summary_writer)
            if not self.quantize_delay:
                ma = self.eval_one_epoch(self.sess)

                # Save the variables to disk.

                if ma > max_accuracy:
                    save_path = self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"))
                    print("Model saved in file: %s" % save_path)
                    max_accuracy = ma
                print("Current model mean accuracy: {}".format(ma))
                print("Best model mean accuracy: {}".format(max_accuracy))
            else:
                if epoch % 5 == 0:
                    if self.checkpoint_path:
                        save_path = self.saver.save(self.sess, os.path.join(self.log_dir, "model-r-{}.ckpt".format(str(epoch))))
                    else:
                        save_path = self.saver.save(self.sess, os.path.join(self.log_dir, "model-{}.ckpt".format(str(epoch))))
                    print("Model saved in file: %s" % save_path)


        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_X):
        return self.sess.run(self.prediction,
                             feed_dict={self.pointclouds_pl: data_X,
                                        self.is_training: False})

    def run_inference(self, sess):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self.num_classes)]
        total_correct_class = [0 for _ in range(self.num_classes)]

        for fn in range(len(TEST_FILES)):
            print('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
            current_data = current_data[:, 0:self.num_points, :]
            # current_data = np.expand_dims(current_data, axis=-2)
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // self.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                feed_dict = {self.pointclouds_pl: current_data[start_idx:end_idx, :, :],
                             self.labels_pl: current_label[start_idx:end_idx],
                             self.is_training: is_training
                             }

                step, pred_val = sess.run([self.batch, self.prediction], feed_dict=feed_dict)
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum(pred_val == current_label[start_idx:end_idx])
                total_correct += correct
                total_seen += self.batch_size
                for i in range(start_idx, end_idx):
                    l = current_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i - start_idx] == l)

        acc = total_correct / float(total_seen)
        print('eval accuracy: %f' % acc)
        aca = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        print('eval avg class acc: %f' % aca)

    def load_model_from_file(self, load_path):
        # ckpt = tf.train.get_checkpoint_state(load_path)
        if load_path and tf.train.checkpoint_exists(load_path):
            print("Reading model parameters from %s" % load_path)
            self.saver.restore(self.sess, load_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())


class SmallModel:
    def __init__(self, args, model_type):
        self.learning_rate = args.learning_rate
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.num_points = 1024
        self.num_classes = 40
        self.checkpoint_path = args.checkpoint_path
        self.MODEL = importlib.import_module(args.model)  # import network module
        self.quantize_delay = args.quantize_delay_s
        self.DYNAMIC = True if args.dynamic_s < 0 else False
        self.STN = True if args.stn_s < 0 else False
        self.SCALE = args.scale_s
        self.CONCAT = True if args.concat_s == 1 else False
        self.BN_DECAY_DECAY_STEP = float(args.decay_step)

        self.temperature = args.temperature
        self.checkpoint_file = "smallmodel"
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type

        self.build_model()

        self.saver = tf.train.Saver()

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,  # Base learning rate.
            batch * self.batch_size,  # Current index into the dataset.
            self.BN_DECAY_DECAY_STEP,  # Decay step.
            0.7,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
        return learning_rate

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch * self.batch_size,
            self.BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_momentum = tf.maximum(BN_DECAY_CLIP, bn_momentum)
        return bn_momentum
        # bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        # return bn_decay

    # Create model
    def build_model(self):
        self.pointclouds_pl = self.MODEL.placeholder_input(self.batch_size, self.num_points)
        self.labels_pl = self.MODEL.placeholder_label(self.batch_size)
        if not self.quantize_delay:
            self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        else:
            self.is_training = True

        self.flag = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
        self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes],
                                     name="%s_%s" % (self.model_type, "softy"))
        self.softmax_temperature = tf.placeholder(tf.float32,
                                                  name="%s_%s" % (self.model_type, "softmaxtemperature"))

        # Note the global_step=batch parameter to minimize.
        # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
        self.batch = tf.Variable(0)
        # bn_decay = BN_INIT_DECAY
        bn_decay = self.get_bn_decay(self.batch)

        # Get model
        with tf.name_scope("%s" % (self.model_type)), tf.variable_scope("%s" % (self.model_type)):

            logits, end_points = self.MODEL.get_network(self.pointclouds_pl, self.is_training,
                                                        bn_decay=bn_decay,
                                                        dynamic=self.DYNAMIC,
                                                        STN=self.STN,
                                                        scale=self.SCALE,
                                                        concat_fea=self.CONCAT)

        with tf.name_scope("%sprediction" % (self.model_type)), tf.variable_scope("%sprediction" % (self.model_type)):
            self.prediction = tf.nn.softmax(logits)
            # Evaluate model
            correct = tf.equal(tf.argmax(logits, 1), tf.cast(self.labels_pl, tf.int64))
            self.accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self.batch_size)

        with tf.name_scope("%soptimization" % (self.model_type)), tf.variable_scope(
            "%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.standard_loss = self.MODEL.get_loss(logits, self.labels_pl, end_points)
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.model_type)
            all_losses = []
            all_losses.append(self.standard_loss)
            all_losses.append(tf.add_n(regularization_losses))
            self.total_loss = tf.add_n(all_losses)

            self.loss_op_soft = tf.cond(self.flag,
                                        true_fn=lambda: tf.nn.l2_loss(self.soft_Y - logits) / self.batch_size,
                                        false_fn=lambda: 0.0)
            self.total_loss += tf.square(self.softmax_temperature) * self.loss_op_soft

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_type)
            learning_rate = self.get_learning_rate(self.batch)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            with tf.control_dependencies([tf.group(*update_ops)]):
                self.train_op = optimizer.minimize(self.total_loss, global_step=self.batch)

            tf.summary.scalar('loss', self.total_loss)
            tf.summary.scalar('standard loss', self.standard_loss)
            tf.summary.scalar('soft loss', self.loss_op_soft)

        with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
            "%ssummarization" % (self.model_type)):
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('bn_decay', bn_decay)
            tf.summary.scalar("loss", self.total_loss)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Merge all summaries into a single op

            # If using TF 1.6 or above, simply use the following merge_all function
            # which supports scoping
            # self.merged_summary_op = tf.summary.merge_all(scope=self.model_type)

            # Explicitly using scoping for TF versions below 1.6

            def mymergingfunction(scope_str):
                with tf.name_scope("%s_%s" % (self.model_type, "summarymerger")), tf.variable_scope(
                    "%s_%s" % (self.model_type, "summarymerger")):
                    from tensorflow.python.framework import ops as _ops
                    key = _ops.GraphKeys.SUMMARIES
                    summary_ops = _ops.get_collection(key, scope=scope_str)
                    if not summary_ops:
                        return None
                    else:
                        return tf.summary.merge(summary_ops)

            self.merged_summary_op = mymergingfunction(self.model_type)

    def start_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)

    def close_session(self):
        self.sess.close()

    def train(self, teacher_model=None):
        teacher_flag = False
        if teacher_model is not None:
            teacher_flag = True

        def train_one_epoch(sess, train_writer):
            """ ops: dict mapping from string to tf ops """
            is_training = True

            # Shuffle train files
            train_file_idxs = np.arange(0, len(TRAIN_FILES))
            np.random.shuffle(train_file_idxs)

            mean_acc = 0
            for fn in range(len(TRAIN_FILES)):
                print('----' + str(fn) + '-----')
                current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
                current_data = current_data[:, 0:self.num_points, :]
                current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
                current_label = np.squeeze(current_label)

                file_size = current_data.shape[0]
                num_batches = file_size // self.batch_size

                total_correct = 0
                total_seen = 0
                loss_sum = 0

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = (batch_idx + 1) * self.batch_size

                    # Augment batched point clouds by rotation and jittering
                    rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
                    jittered_data = provider.jitter_point_cloud(rotated_data)
                    jittered_data = provider.random_scale_point_cloud(jittered_data)
                    jittered_data = provider.rotate_perturbation_point_cloud(jittered_data)
                    jittered_data = provider.shift_point_cloud(jittered_data)

                    soft_targets = current_label[start_idx:end_idx]
                    n_values = np.zeros((soft_targets.size, self.num_classes))
                    n_values[np.arange(soft_targets.size), soft_targets] = 1
                    soft_targets = n_values
                    if teacher_flag:
                        soft_targets = teacher_model.predict(current_data[start_idx:end_idx, :, :])

                    # print(soft_targets, current_label[start_idx:end_idx])

                    if not self.quantize_delay:
                        feed_dict = {
                            # ops['pointclouds_pl']: jittered_data,
                            self.pointclouds_pl: current_data[start_idx:end_idx, :, :],
                            self.labels_pl: current_label[start_idx:end_idx],
                            self.is_training: is_training,
                            self.soft_Y: soft_targets,
                            self.flag: teacher_flag,
                            self.softmax_temperature: self.temperature
                        }
                    else:
                        feed_dict = {
                            # ops['pointclouds_pl']: jittered_data,
                            self.pointclouds_pl: current_data[start_idx:end_idx, :, :],
                            self.labels_pl: current_label[start_idx:end_idx],
                            self.soft_Y: soft_targets,
                            self.flag: teacher_flag,
                            self.softmax_temperature: self.temperature
                            # ops['is_training_pl']: is_training,
                        }

                    # print(feed_dict)
                    summary, step, _, loss_val, pred_val = sess.run([self.merged_summary_op, self.batch, self.train_op,
                                                                     self.total_loss, self.prediction],
                                                                    feed_dict=feed_dict)
                    # sess.run(ops['mask_update_op'])
                    train_writer.add_summary(summary, step)
                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])
                    total_correct += correct
                    total_seen += self.batch_size
                    loss_sum += loss_val

                print('mean loss: %f' % (loss_sum / float(num_batches)))
                print('accuracy: %f' % (total_correct / float(total_seen)))

                mean_acc += total_correct / float(total_seen)
            return mean_acc / len(TRAIN_FILES)

        def eval_one_epoch(sess):
            """ ops: dict mapping from string to tf ops """
            is_training = False
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(self.num_classes)]
            total_correct_class = [0 for _ in range(self.num_classes)]

            for fn in range(len(TEST_FILES)):
                print('----' + str(fn) + '-----')
                current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
                current_data = current_data[:, 0:self.num_points, :]
                # current_data = np.expand_dims(current_data, axis=-2)
                current_label = np.squeeze(current_label)

                file_size = current_data.shape[0]
                num_batches = file_size // self.batch_size

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = (batch_idx + 1) * self.batch_size

                    feed_dict = {self.pointclouds_pl: current_data[start_idx:end_idx, :, :],
                                 self.labels_pl: current_label[start_idx:end_idx],
                                 self.soft_Y: [[0] * self.num_classes] * self.batch_size,
                                 self.is_training: is_training,
                                 self.flag: False,
                                 self.softmax_temperature: 1.0
                                 }

                    summary, step, loss_val, pred_val = sess.run([self.merged_summary_op, self.batch,
                                                                  self.standard_loss, self.prediction],
                                                                 feed_dict=feed_dict)
                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])
                    total_correct += correct
                    total_seen += self.batch_size
                    loss_sum += (loss_val * self.batch_size)
                    for i in range(start_idx, end_idx):
                        l = current_label[i]
                        total_seen_class[l] += 1
                        total_correct_class[l] += (pred_val[i - start_idx] == l)

            print('eval mean loss: %f' % (loss_sum / float(total_seen)))
            acc = total_correct / float(total_seen)
            print('eval accuracy: %f' % acc)
            aca = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            print('eval avg class acc: %f' % aca)

            return acc

        # Initialize the variables (i.e. assign their default value)
        self.sess.run(tf.global_variables_initializer())

        print("Starting Training")

        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        max_accuracy = 0
        if self.checkpoint_path:
            print("restore model parameters from {}".format(self.checkpoint_path))
            scope_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_type))
            scope_saver.restore(self.sess, self.checkpoint_path)
        for epoch in range(self.num_epoch + 1):
            print(('**** EPOCH %03d ****' % (epoch))
                  + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '****')
            ma = train_one_epoch(self.sess, train_summary_writer)
            if not self.quantize_delay:
                ma = eval_one_epoch(self.sess)

                # Save the variables to disk.

                if ma > max_accuracy:
                    save_path = self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"))
                    print("Model saved in file: %s" % save_path)
                    max_accuracy = ma
                print("Current model mean accuracy: {}".format(ma))
                print("Best model mean accuracy: {}".format(max_accuracy))
            else:
                if epoch % 5 == 0:
                    if self.checkpoint_path:
                        save_path = self.saver.save(self.sess, os.path.join(self.log_dir,
                                                                            "model-r-{}.ckpt".format(str(epoch))))
                    else:
                        save_path = self.saver.save(self.sess,
                                                    os.path.join(self.log_dir, "model-{}.ckpt".format(str(epoch))))
                    print("Model saved in file: %s" % save_path)

        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.pointclouds_pl: data_X,
                                        self.flag: False, self.softmax_temperature: temperature})

    # def run_inference(self, dataset):
    #     test_images, test_labels = dataset.get_test_data()
    #     print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_images,
    #                                                                        self.Y: test_labels,
    #                                                                        # self.soft_Y: test_labels,
    #                                                                        self.flag: False,
    #                                                                        self.softmax_temperature: 1.0
    #                                                                        }))

    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())