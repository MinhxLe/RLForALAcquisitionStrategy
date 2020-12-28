"""
al run managers a single AL session. It handles logging
"""
import csv
import numpy as np

import os
import time
import tensorflow as tf

from attr import attrs, attrib
from datetime import datetime
from src.environment import ClassiferALEnvironmentT
from src.al_agent import ClassifierALAgentT
from src.utils.log_utils import (
    set_up_experiment_logging,
    time_display,
)
from sklearn.metrics import f1_score, confusion_matrix


@attrs
class ClassiferALSessionManager:
    al_agent: ClassifierALAgentT = attrib()
    al_env: ClassiferALEnvironmentT = attrib()
    al_manager = attrib()
    session_dir: str = attrib()

    al_epochs: int = attrib()
    al_step_percentage: float = attrib()
    warm_start_percentage: float = attrib(default=0)
    retrain_model: bool = attrib(default=False)

    save_model_interval:int = attrib(default=10)
    stdout: bool = attrib(default=False)

    # only made available when init_session is called
    start_time:int = None
    run_dir:str = None
    logger = None
    tf_summary_writer = None
    model_snapshot_dir: str = None

    def reset_session(self):
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.run_dir = os.path.join(self.session_dir, timestamp_str)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        # setting logs, tf sumary writer and some
        self.logger, self.tf_summary_writer, self.model_snapshot_dir = (
            set_up_experiment_logging(
                self.run_dir,
                log_fpath=os.path.join(self.run_dir, "session.log"),
                model_snapshot_dir=os.path.join(self.run_dir, "model_snapshots"),
                metrics_dir=os.path.join(self.run_dir, "metrics"),
                stdout=self.stdout,
                clear_old_data=True,
            )
        )
        self.al_env.reset()
        self.start_time = None

    def run_session(self):
        pool_size = self.al_manager.pool_size
        n_points_to_label = int(pool_size*self.al_step_percentage)

        self.start_time = time.monotonic()
        self.logger.info(f"Starting session in f{self.run_dir}")
        # warm start
        if self.warm_start_percentage > 0:
            warm_start_count = int(self.warm_start_percentage * pool_size)
            self.logger.info(f"Warm start of {warm_start_count} labels")
            self.al_env.warm_start(warm_start_count)
            self.al_env.train_step()

        for al_epoch in range(0, self.al_epochs):
            n_step = self.al_env.n_step
            self.logger.info("-" * 118)
            self.logger.info(
                    f"AL Epoch: {al_epoch+1}/{self.al_epochs}"
                    f"\tTrain Data Labeled: {n_step}/{pool_size}"
                    f"\tElapsed Time: {time_display(time.monotonic()-self.start_time)}")

            # label step
            selection = self.al_agent.select_data_to_label(n_points_to_label)
            # TODO add metrics around selection
            self.al_env.label_step(selection)

            self.al_env.train_step(retrain=self.retrain_model)

            self.log_metrics(n_step, "train")
            self.log_metrics(n_step, "test")
            self.log_metrics(n_step, "validation")

            # save model
            if (self.save_model_interval > 0 and
                ((al_epoch+1) % self.save_model_interval == 0)):
                model_fpath = os.path.join(
                    self.model_snapshot_dir,
                    f"model_AL_epoch_{al_epoch}_{self.al_epochs}.ckpt")
                self.al_env.model_manager.save_model(model_fpath)

    def log_metrics(
            self,
            step: int,  # training step (number data point labeled)
            data_type: str,  # test, train, validation
            ):
        """
        evaluates model and input, prediction, and true label
        """
        x, y = self.al_manager.get_dataset(data_type)

        # TODO rely on model compiling here?
        loss_metric = tf.keras.metrics.Mean(name="loss")
        micro_f1_metric = tf.keras.metrics.Mean(name="micro_f1_metric")
        macro_f1_metric = tf.keras.metrics.Mean(name="macro_f1_metric")

        loss_metric.reset_states()
        micro_f1_metric.reset_states()
        macro_f1_metric.reset_states()
        model_num_classes = None
        total_prediction_count = None
        total_true_label_count = None
        cm = None


        for batch_x, batch_y, raw_prediction, batch_loss in \
                self.al_env.model_manager.evaluate_model(x, y):

            loss_metric.update_state(batch_loss)

            # dynamically getting number of class
            model_num_classes = raw_prediction.shape[-1]
            if total_prediction_count is None:
                total_prediction_count = np.zeros(model_num_classes)
                total_true_label_count = np.zeros(model_num_classes)

            prediction = np.argmax(raw_prediction, axis=1)
            unique, counts = np.unique(prediction, return_counts=True)
            for i, count in zip(unique, counts):
                total_prediction_count[i] += count

            total_true_label_count += np.sum(batch_y, axis=0)
            batch_y = np.argmax(batch_y, axis=1)  # 1 hot to class
            if cm is None:
                cm = confusion_matrix = confusion_matrix(batch_y, prediction)
            else:
                cm += confusion_matrix(batch_y, prediction)
            micro_f1_metric.update_state(
                f1_score(batch_y, prediction, average="micro", labels=np.arange(model_num_classes)))
            macro_f1_metric.update_state(
                f1_score(batch_y, prediction, average="macro", labels=np.arange(model_num_classes)))

        # min class ratio
        min_class_prediction_ratio = np.min(total_prediction_count)/data_size
        min_class_true_ratio = np.min(total_true_label_count)/data_size

        # max class ratio
        max_class_prediction_ratio = np.max(total_prediction_count)/data_size
        max_class_true_ratio = np.max(total_true_label_count)/data_size

        metrics = {
            "loss": loss_metric.result(),
            "micro_f1_metric": micro_f1_metric.result(),
            "macro_f1_metric": macro_f1_metric.result(),
            "min_class_prediction_ratio": min_class_prediction_ratio,
            "min_class_true_ratio": min_class_true_ratio,
            "max_class_prediction_ratio": max_class_prediction_ratio,
            "max_class_true_ratio": max_class_true_ratio,
        }

        # tensorflow metric output
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

        # dumping confusion_matrix
        cm = cm.flatten()
        data = np.array([step]) + cm
        cm_file = os.path.join(self.run_dir, f"{data_type}_confusion_matrix.csv")
        with open(cm_file, 'a') as f:
            np.savetxt(f, data, delimiter=",")
