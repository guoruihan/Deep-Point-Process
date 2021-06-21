
import os
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


#
# the plot function is copied from [ https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html ]
#
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


class Logger:
    def __init__(self, dir, args):
        self.global_step = 0
        self.local_step = 0
        self.evaluation_id = 0

        run = 0
        while os.path.exists(f"{dir}/run{run}"):
            run += 1
        dir = f"{dir}/run{run}"
        self.dir = dir
        self.cm_dir = f'{dir}/cm_dir'
        os.makedirs(dir)
        os.makedirs(self.cm_dir)
        self.writer = SummaryWriter(log_dir=dir)
        with open(f'{dir}/args.json', 'w') as f:
            json.dump(args, f, indent=4)

    def log_new_epoch(self, epoch):
        print(f"Epoch {epoch}")
        self.local_step = 0

    def log_train(self, time_loss, event_loss, merged_loss):
        self.global_step += 1
        self.local_step += 1
        print(f"{self.local_step} {self.global_step} Time loss: {time_loss:8.3f} Event loss: {event_loss:8.3f} Merged loss: {merged_loss:8.3f}")
        self.writer.add_scalars('loss', {'time': time_loss, 'event': event_loss, 'merged': merged_loss}, global_step=self.global_step)

    def log_evaluation(self, evaluation_result, is_test):
        self.evaluation_id += 1
        target_time, predicted_time, target_events, predicted_events = evaluation_result

        tmp1 = map(abs, np.array(target_time) - np.array(predicted_time))
        tmp2 = map(abs, np.array(target_events) - np.array(predicted_events))
        print("average_time_diff", sum(tmp1) / len(target_time))
        print("average_event_diff", sum(tmp2) / len(target_time))

        return
