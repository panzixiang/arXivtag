from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools
import numpy as np
import sys
import csv


def main():
    test_label = [1]*253 + [2]*391 + [3]*387 + [4]*409 + [5]*867 + \
        [6]*381 + [7]*41 + [8]*18 + [9]*132 + [10]*48

    filename = sys.argv[1]

    with open(filename, 'rU') as f:
        pred = [rec for rec in csv.reader(f, delimiter=',')]
    pred = sum(pred, [])
    pred = [int(x) for x in pred]
    print("accuracy: " + str(accuracy_score(pred, test_label)))
    cm = confusion_matrix(test_label, pred, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    np.set_printoptions(precision=2)
    # normalize and convert to pct
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.around(cm, decimals=1)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plot_confusion_matrix(cm)
    plt.show()
    # plt.imasve('smv_confusion', cm, cmap=plt.cm.viridis)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap="inferno"):
    hfont = {'fontname': 'Sans'}
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20, **hfont)
    plt.colorbar()
    tick_marks = np.arange(10)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.xticks(tick_marks, ['astro', 'cond', 'cs', 'hep', 'math', 'physics', 'q-bio', 'q-fin', 'quant', 'stat'],
               rotation=45, fontsize=16, **hfont)
    plt.yticks(tick_marks, ['astro', 'cond', 'cs', 'hep', 'math', 'physics', 'q-bio', 'q-fin', 'quant', 'stat'],
               fontsize=16, **hfont)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=16, **hfont)
    plt.xlabel('Predicted label', fontsize=16, **hfont)


if __name__ == "__main__":
    main()
