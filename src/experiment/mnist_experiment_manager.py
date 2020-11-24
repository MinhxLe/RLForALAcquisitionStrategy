import abc
import os
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

from src.model.mnist_model import MNISTModel
from src.sampler import (
    ALRandomSampler,
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

        if not os.path.exists(args.experiment_dir):
            os.makedirs(args.experiment_dir)

        self.logger, self.tf_summary_writer, self.model_snapshot_dir = (
            set_up_experiment_logging(
                args.experiment_name,
                log_fpath=os.path.join(args.experiment_dir, "experiment.log"),
                model_snapshot_dir=os.path.join(args.experiment_dir, "model_snapshots"),
                metrics_dir=os.path.join(args.experiment_dir, "metrics"),
                stdout=args.stdout,
                clear_old_data=True,
            )
        )
        self.start_time = None
        self._init_data()
        self.model = self._get_model()
        self.optimizer = self._get_optimizer()
        self.loss_fn = self._get_loss()
        # seed random
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    def _init_data(self) -> None:
        # loads from keras cache
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # normalizes the features
        x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

        # changing to binary classifier
        y_train = np.expand_dims((y_train==9).astype(np.int32), 1)
        y_test = np.expand_dims((y_test==9).astype(np.int32), 1)

        # TODO fix
        if self.args.debug:
            x_train = x_train[:100]
            y_train = y_train[:100]
            y_train[-50:] = 1
            x_test = x_test[:100]
            y_test = y_test[:100]
            y_test[-50:] = 1

        # we keep as raw numpy as it's easier to index only the labelled set
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)

    def _get_model(self) -> tf.keras.Model:
        args = self.args
        return MNISTModel(
            args.model_num_filters,
            args.model_filter_size,
            args.model_pool_size,
        )

    def _get_optimizer(self):
        # TODO add args
        return tf.keras.optimizers.Adam()

    def _get_loss(self):
        # TODO switch to BinaryCrossentropy
        # weight the class(?)
        return tf.keras.losses.BinaryCrossentropy(
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

        loss_metric = tf.keras.metrics.Mean(name="loss")
        f1_metric = tf.keras.metrics.Mean(name="f1_metric")
        prediction_class_ratio_metric = tf.keras.metrics.Mean(name="prediction_ratio")

        test_x, test_y = data
        data_size = len(test_x)
        total_f1_score = 0
        total_batch_count = 0
        total_num_positive = 0

        for idx, test_batch in enumerate(
                batch_sample_indices(data_size, batch_size=args.batch_size)):
            batch_x, batch_y = test_x[test_batch], test_y[test_batch]
            raw_prediction = model(batch_x, training=False)
            loss_metric(loss_fn(batch_y, raw_prediction))

            prediction = np.int32(raw_prediction>=0.5)
            prediction_class_ratio_metric(prediction)
            f1_metric(f1_score(batch_y, prediction, average="binary"))

        return {
            "loss": loss_metric.result(),
            "f1_score": f1_metric.result(),
            "class_prediction_ratio": prediction_class_ratio_metric.result(),
        }


    def run_experiment(self):
        self.start_time = time.monotonic()
        start_time = self.start_time
        args = self.args
        train_data = self.train_data
        test_data = self.test_data
        logger = self.logger
        AL_sampler = self._get_sampler()

        labelled_indices = set()

        logger.info(f"Starting {args.experiment_name} experiment")
        n_labels = len(train_data[0])//args.al_epochs
        train_dataset_size = len(train_data[0])
        try:
            # TODO bug of missing final count of labelled
            for curr_AL_epoch in range(args.al_epochs):
                # AL step
                self.label_n_elements(AL_sampler, n_labels)
                labelled_indices = AL_sampler.labelled_idx_set
                n_labeled = len(labelled_indices)

                logger.info("-" * 118)
                logger.info(
                        f"AL Epoch: {curr_AL_epoch+1}/{args.al_epochs}"
                        f"\tTrain Data Labeled: {n_labeled}/{train_dataset_size}"
                        f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
                # train step
                train_data = self.get_labelled_train_data(AL_sampler)
                for epoch in range(1, args.train_epochs+1):
                    logger.info(f"Train Epoch: {epoch}/{args.train_epochs}"
                                f"\tElapsed Time: {time_display(time.monotonic()-start_time)}")
                    self.train_model_step(train_data)

                # final train loss (full data)
                metrics = self.evaluate_model_step(train_data)
                with self.tf_summary_writer.as_default():

                    for metric_key, metric_value in metrics.items():
                        tf.summary.scalar(f"train {metric_key}", metric_value, step=n_labeled)

                # validation step
                metrics = self.evaluate_model_step(test_data)
                log_str = (
                    f"AL Epoch:{curr_AL_epoch}/{args.al_epochs}"
                    f"\tTrain Data Labeled: {n_labeled}/{train_dataset_size}")
                for metric_key, metric_value in metrics.items():
                    log_str += f"\tTest {metric_key}: {metric_value}"
                log_str += f"\tElapsed Time: {time_display(time.monotonic()-start_time)}"

                logger.info(log_str)
                with self.tf_summary_writer.as_default():
                    for metric_key, metric_value in metrics.items():
                        tf.summary.scalar(f"test {metric_key}", metric_value, step=n_labeled)
                # save model
                if (args.save_model_interval > 0 and
                    ((curr_AL_epoch+1) % args.save_model_interval == 0)):
                    model_fpath = os.path.join(
                        self.model_snapshot_dir,
                        f"model_AL_epoch_{curr_AL_epoch}_{args.al_epochs}.ckpt")
                    self.model.save_weights(model_fpath)
        except KeyboardInterrupt:
            logger.warning('Exiting from training early!')

        # TODO add train losses
        # TODO extract out
        # n = np.min((len(labelled_data_counts), len(test_f1_scores), len(test_losses)))
        # results_df = pd.DataFrame.from_dict({
        #     "labelled_data_counts": labelled_data_counts[:n],
        #     "test_f1_scores": test_f1_scores[:n],
        #     "test_losses": test_losses[:n]})
        # results_df.to_csv(os.path.join(self.experiment_dir, "test_results.csv"))


class ALRandomExperimentManager(ActiveLearningExperimentManagerT):

    def _get_sampler(self):
        return ALRandomSampler(len(self.train_data[0]))

    def label_n_elements(self, sampler, n_elements):
        sampler.label_n_elements(n_elements)

# class MNLPExperimentManager(ActiveLearningExperimentManagerT):

#     def _get_sampler(self):
#         return MNLPSampler(self.train_data)

#     def label_n_elements(self, sampler, n_elements):
#         self.sampler.label_n_elements(n_elements, self.model)
