import abc
import csv
import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score

from src.model.mnist_model import MNISTModel
from src.model.cifar10_model import Cifar10Model
from src.sampler import (
    ALRandomSampler,
    LeastConfidenceSampler,
    UCBBanditSampler
)
from src.utils.log_utils import (
    set_up_experiment_logging,
    time_display,
)
from src.utils.utils import (
    batch_sample_indices,
)


class ActiveLearningExperimentManagerT(abc.ABC):
    """
    - handles managing data set
    - interface between model and raw data
    - logging
    - model generation
    """
    def __init__(self, args):
        self.args = args
        # these don't change run to run
        self._init_data()
        self.optimizer = self._get_optimizer()
        self.loss_fn = self._get_loss()

        # we only seed once
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

        # these are none until you call _init_experiment
        self.logger = None
        self.tf_summary_writer = None
        self.model = None
        self.start_time = None
        self.run_dir = None
        self.AL_sampler = None

    def _init_data(self) -> None:
        args = self.args
        if args.dataset == "mnist":
            # loads from keras cache
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            # normalizes the features
            x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

            # change to 1 hot
            y_train = np.eye(args.model_num_classes)[y_train]
            y_test = np.eye(args.model_num_classes)[y_test]
        elif args.dataset == "cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

            # Normalize pixel values to be between 0 and 1
            x_train, x_test = x_train / 255.0, x_test / 255.0
            y_train = np.eye(10)[y_train.flatten()]
            y_test = np.eye(10)[y_test.flatten()]
        else:
            raise NotImplementedError("experiment for dataset not supported")

        def build_rare_class_dataset(data, rare_class, rare_class_percentage):
            x, y = data
            # keep nonrare class data as is
            x_non_rare = x[y!=rare_class]
            y_non_rare = y[y!=rare_class]

            # in test and training, we make 1 of the classes really rare
            unique, counts = np.unique(y, return_counts=True)
            rare_class_count = counts[unique==rare_class]
            count_to_keep = int(rare_class_count * rare_class_percentage)
            x_rare = x[y==rare_class][:count_to_keep]
            y_rare = y[y==rare_class][:count_to_keep]

            return (np.concatenate((x_non_rare, x_rare)),
                    np.concatenate((y_non_rare, y_rare)))

        if args.rare_class:
            x_train, y_train = build_rare_class_dataset(
                (x_train, y_train),
                args.rare_class,
                args.rare_class_percentage)

            x_test, y_test = build_rare_class_dataset(
                (x_test, y_test),
                args.rare_class,
                args.rare_class_percentage)
        if self.args.debug:
            n_points = 1000
            idx = np.random.choice(
                len(x_train), n_points)
            x_train = x_train[idx]
            y_train = y_train[idx]
            idx = np.random.choice(
                len(x_test), n_points)
            x_test = x_test[idx]
            y_test = y_test[idx]

        num_validation = int(len(x_train) * args.validation_percentage)
        idx = np.random.choice(len(x_train), num_validation)
        mask = np.ones(len(x_train), np.bool)
        mask[idx] = 0

        x_val = x_train[~mask]
        y_val = y_train[~mask]
        x_train = x_train[mask]
        y_train = y_train[mask]

        # we keep as raw numpy as it's easier to index only the labelled set
        self.train_data = (x_train, y_train)
        self.val_data = (x_train, y_train)
        self.test_data = (x_test, y_test)

    def _get_model(self) -> tf.keras.Model:
        args = self.args
        if args.dataset == "mnist":
            return MNISTModel(
                args.model_num_filters,
                args.model_filter_size,
                args.model_pool_size,
                args.model_num_classes,
            )
        elif args.dataset == "cifar10":
            return Cifar10Model()
        else:
            raise NotImplementedError("model for dataset not implemented")

    def _get_optimizer(self):
        # TODO add args
        return tf.keras.optimizers.Adam()

    def _get_loss(self):
        # TODO switch to BinaryCrossentropy
        # weight the class(?)
        # return tf.keras.losses.BinaryCrossentropy(
        #     from_logits=True,
        #     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        return tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    @abc.abstractmethod
    def _get_sampler(self):
        raise NotImplementedError("pick a sampler")

    @abc.abstractmethod
    def label_n_elements(self, sampler, n_elements):
        raise NotImplementedError("using sampler not implemented")

    def get_labelled_train_data(self, sampler):
        labelled_indices = list(sampler.labelled_idx_set)
        train_x = self.train_data[0][labelled_indices]
        train_y = self.train_data[1][labelled_indices]
        return (train_x, train_y)

    def train_model_step(self, data):
        """
        single pass of labelled train data
        """
        # TODO we don't retrain here, is that sensible?
        model = self.model
        args = self.args
        logger = self.logger
        optimizer = self.optimizer
        loss_fn = self.loss_fn

        total_loss = 0
        count = 0
        elapsed = None
        train_x, train_y = data
        data_size = len(train_x)
        for idx, batch in enumerate(batch_sample_indices(data_size, batch_size=args.batch_size)):
            batch_x, batch_y = train_x[batch], train_y[batch]
            with tf.GradientTape() as tape:
                prediction = model(batch_x)
                loss = loss_fn(batch_y, prediction)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()
            count += len(batch_x)

            if self.start_time:
                elapsed = time.monotonic() - self.start_time
            if args.train_log_interval > 0 and (idx+1) % args.train_log_interval == 0:
                cur_loss = total_loss / count
                logger.info(f"Batch: {(idx+1)*args.batch_size}/{data_size}"
                        f"\tLoss: {cur_loss}"
                        f"\tElapsed Time: {time_display(elapsed)}")
                total_loss = 0
                count = 0

    def evaluate_model_step(self, data):
        args = self.args
        model = self.model
        logger = self.logger
        optimizer = self.optimizer
        loss_fn = self.loss_fn

        # TODO rely on model compiling here?
        loss_metric = tf.keras.metrics.Mean(name="loss")
        micro_f1_metric = tf.keras.metrics.Mean(name="micro_f1_metric")
        macro_f1_metric = tf.keras.metrics.Mean(name="macro_f1_metric")
        accuracy_metric = tf.keras.metrics.Accuracy(name="accuracy_metric")
        if args.rare_class:
            rare_class_f1_metric = tf.keras.metrics.Mean(name="rare_f1_metric")

        test_x, test_y = data
        data_size = len(test_x)

        total_prediction_count = np.zeros(args.model_num_classes)
        total_true_label_count = np.zeros(args.model_num_classes)
        rare_class_count = 0
        for idx, test_batch in enumerate(
                batch_sample_indices(data_size, batch_size=args.batch_size)):
            batch_x, batch_y = test_x[test_batch], test_y[test_batch]
            raw_prediction = model(batch_x, training=False)
            loss_metric(loss_fn(batch_y, raw_prediction))

            # min class ratio
            min_class_prediction_ratio = np.min(total_prediction_count)/data_size
            min_class_true_ratio = np.min(total_true_label_count)/data_size

            # max class ratio
            max_class_prediction_ratio = np.max(total_prediction_count)/data_size
            max_class_true_ratio = np.max(total_true_label_count)/data_size

            prediction = np.argmax(raw_prediction, axis=1)
            unique, counts = np.unique(prediction, return_counts=True)
            for i, count in zip(unique, counts):
                total_prediction_count[i] += count

            batch_y = np.argmax(batch_y, axis=1)  # 1 hot to class
            unique, counts = np.unique(batch_y, return_counts=True)
            for i, count in zip(unique, counts):
                total_true_label_count[i] += count

            micro_f1_metric(
                f1_score(batch_y, prediction, average="micro", labels=np.arange(10)))
            macro_f1_metric(
                f1_score(batch_y, prediction, average="macro", labels=np.arange(10)))
            accuracy_metric(batch_y, prediction)
            if args.rare_class:
                rare_class_f1_metric(
                    f1_score(batch_y, prediction, average=None, labels=np.arange(10))[args.rare_class])
                rare_class_count += len(batch_y[batch_y==args.rare_class])


        result = {
            "loss": loss_metric.result(),
            "micro_f1_metric": micro_f1_metric.result(),
            "macro_f1_metric": macro_f1_metric.result(),
            "accuracy_metric": accuracy_metric.result(),
            "min_class_prediction_ratio": min_class_prediction_ratio,
            "min_class_true_ratio": min_class_true_ratio,
            "max_class_prediction_ratio": max_class_prediction_ratio,
            "max_class_true_ratio": max_class_true_ratio,

        }
        if args.rare_class:
            # rare class ratio
            rare_class_prediction_ratio = total_prediction_count[args.rare_class]/data_size
            rare_class_true_ratio = total_true_label_count[args.rare_class]/data_size
            result.update({
                "rare_class_f1_metric": rare_class_f1_metric.result(),
                "rare_class_prediction_ratio": rare_class_prediction_ratio,
                "rare_class_true_ratio": rare_class_true_ratio,
                "rare_class_count": rare_class_count,
            })
        return result

    def log_metrics(
            self,
            metrics,
            step: int,  # training step (number data point labeled)
            data_type: str,  # test, train
            ):
        # we copy to make mit immutable
        metrics = metrics.copy()
        # metric output
        with self.tf_summary_writer.as_default():
            for metric_key, metric_value in metrics.items():
                tf.summary.scalar(f"{data_type} {metric_key}", metric_value, step=step)
        # log output
        for metric_key, metric_value in metrics.items():
            self.logger.info(f"{data_type} {metric_key}: {metric_value}")

        # log output, tensorboard output, save to a master csv
        # TODO keep this open for faster run
        metrics["step"] = step
        # we cast to float before storing to csv
        for metric_key, metric_value in metrics.items():
            metrics[metric_key] = float(metric_value)
        csv_file = os.path.join(self.run_dir, f"{data_type}_results.csv")
        with open(csv_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)


    def run_experiment(self):
        try:
            for _ in range(self.args.n_experiment_runs):
                self._init_experiment()
                self._run_experiment()
        except KeyboardInterrupt:
            self.logger.warning('Exiting from training early!')

    def _init_experiment(self):
        args = self.args

        #using timestamp as unique identifier for run
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.run_dir = os.path.join(args.experiment_dir, timestamp_str)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        # setting logs, tf sumary writer and some
        self.logger, self.tf_summary_writer, self.model_snapshot_dir = (
            set_up_experiment_logging(
                args.experiment_name,
                log_fpath=os.path.join(self.run_dir, "experiment.log"),
                model_snapshot_dir=os.path.join(self.run_dir, "model_snapshots"),
                metrics_dir=os.path.join(self.run_dir, "metrics"),
                stdout=args.stdout,
                clear_old_data=True,
            )
        )
        self.model = self._get_model()
        self.AL_sampler = self._get_sampler()
        self.start_time = time.monotonic()

    def _run_experiment(self):
        start_time = self.start_time
        args = self.args
        train_data = self.train_data
        test_data = self.test_data
        logger = self.logger
        AL_sampler = self.AL_sampler

        labelled_indices = set()

        logger.info(f"Starting {args.experiment_name} experiment")
        n_to_label = int((len(train_data[0]) * args.al_step_percentage))

        train_dataset_size = len(train_data[0])

        # TODO bug of missing final count of labelled
        for curr_AL_epoch in range(args.al_epochs):
            # AL step
            self.label_n_elements(AL_sampler, n_to_label)
            labelled_indices = AL_sampler.labelled_idx_set
            n_labeled = len(labelled_indices)

            logger.info("-" * 118)
            logger.info(
                    f"AL Epoch: {curr_AL_epoch+1}/{args.al_epochs}"
                    f"\tTrain Data Labeled: {n_labeled}/{train_dataset_size}"
                    f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
            # train step
            if args.retrain_model_from_scratch:
                self.model = self._get_model()

            train_data = self.get_labelled_train_data(AL_sampler)
            for epoch in range(1, args.train_epochs+1):
                self.train_model_step(train_data)
            # final train loss (full data)
            train_metrics = self.evaluate_model_step(train_data)
            self.log_metrics(train_metrics, n_labeled, "train")

            # test metrics
            test_metrics = self.evaluate_model_step(test_data)
            self.log_metrics(test_metrics, n_labeled, "test")

            # save model
            if (args.save_model_interval > 0 and
                ((curr_AL_epoch+1) % args.save_model_interval == 0)):
                model_fpath = os.path.join(
                    self.model_snapshot_dir,
                    f"model_AL_epoch_{curr_AL_epoch}_{args.al_epochs}.ckpt")
                self.model.save_weights(model_fpath)

class RandomExperimentManager(ActiveLearningExperimentManagerT):

    def _get_sampler(self):
        return ALRandomSampler(len(self.train_data[0]))

    def label_n_elements(self, sampler, n_elements):
        sampler.label_n_elements(n_elements)


class LCExperimentManager(ActiveLearningExperimentManagerT):

    def _get_sampler(self):
        return LeastConfidenceSampler(self.train_data[0])

    def label_n_elements(self, sampler, n_elements):
        sampler.label_n_elements(n_elements, self.model)


class RLExperimentManagerT(ActiveLearningExperimentManagerT):
    def log_RL_metrics(self, step):
        AL_sampler = self.AL_sampler
        with self.tf_summary_writer.as_default():
            tf.summary.scalar("Action selected", self.action, step=step)
            for i, arm_count in enumerate(AL_sampler.arm_count):
                tf.summary.scalar(
                    f"{AL_sampler.get_action(i)} action count", arm_count, step=step)
        self.logger.info(f"selected {AL_sampler.get_action(self.action)}")
        # TODO add in CSV to log arm + sampler state

    def _init_experiment(self):
        super()._init_experiment()
        self.reward = None # keeps track of reward
        self.state = None  # and state
        self.action = None # track of most recent action

        # these are all thigns relevent to compute ^
        self.val_metrics = None

    @abc.abstractmethod
    def update_state(self):
        """
        updates state of system  and sets it in self.state
        we store reward in part of class since we don't know what we need to keep track of
        in actual experiment
        """
        ...

    @abc.abstractmethod
    def update_reward(self):
        """
        updates reward and set its in self.reward
        we store reward in part of class since we don't know what we need to keep track of
        in actual experiment
        """
        ...

    @abc.abstractmethod
    def update_agent_sampler(self, sampler, action, reward, state):
        # TODO add in batch option (?)
        ...

    def _run_experiment(self):
        start_time = self.start_time
        args = self.args
        train_data = self.train_data
        test_data = self.test_data
        logger = self.logger
        AL_sampler = self.AL_sampler

        labelled_indices = set()

        logger.info(f"Starting {args.experiment_name} experiment")
        n_to_label = int(len(train_data[0]) * args.al_step_percentage)
        train_dataset_size = len(train_data[0])

        # TODO bug of missing final count of labelled
        for curr_AL_epoch in range(args.al_epochs):
            # AL step
            self.action, _ = self.label_n_elements(AL_sampler, n_to_label)
            labelled_indices = AL_sampler.labelled_idx_set
            n_labeled = len(labelled_indices)

            logger.info("-" * 118)
            self.log_RL_metrics(n_labeled)
            logger.info(
                    f"AL Epoch: {curr_AL_epoch+1}/{args.al_epochs}"
                    f"\tTrain Data Labeled: {n_labeled}/{train_dataset_size}"
                    f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")

            # train step
            train_data = self.get_labelled_train_data(AL_sampler)
            for epoch in range(1, args.train_epochs+1):
                self.train_model_step(train_data)

            # final train loss (full data)
            train_metrics = self.evaluate_model_step(train_data)
            self.log_metrics(train_metrics, n_labeled, "train")

            # test metrics
            test_metrics = self.evaluate_model_step(test_data)
            self.log_metrics(test_metrics, n_labeled, "test")

            val_metrics = self.evaluate_model_step(self.val_data)
            self.log_metrics(val_metrics, n_labeled, "val")
            self.val_metrics = val_metrics

            self.update_reward()
            self.update_state()
            self.update_agent_sampler(
                AL_sampler, self.action, self.reward, self.state)

            # TODO save RL agent

            # save model
            if (args.save_model_interval > 0 and
                ((curr_AL_epoch+1) % args.save_model_interval == 0)):
                model_fpath = os.path.join(
                    self.model_snapshot_dir,
                    f"model_AL_epoch_{curr_AL_epoch}_{args.al_epochs}.ckpt")
                self.model.save_weights(model_fpath)

class UCBBanditExperimentManager(RLExperimentManagerT):
    def __init__(self, args):
        assert args.reward_metric_name is not None
        super().__init__(args)

    def _get_sampler(self):
        return UCBBanditSampler(self.train_data[0])

    def label_n_elements(self, sampler, n_elements):
        return sampler.label_n_elements(n_elements, self.model)

    def _init_experiment(self):
        super()._init_experiment()
        self.previous_metric_value = 0

    def update_state(self):
        self.state = None

    def update_reward(self):
        curr_metric_value = self.val_metrics[self.args.reward_metric_name]
        # TODO explore scaling down based on state
        self.reward = curr_metric_value - self.previous_metric_value
        self.previous_metric_value = curr_metric_value

    def update_agent_sampler(self, sampler, action, reward, state):
        self.AL_sampler.update_q_value(self.action, reward)
