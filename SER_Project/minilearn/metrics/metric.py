from .._validation import targets, classification_targets
import numpy as np

def paired_targets(y_t, y_p):
    y_t, y_p = classification_targets(y_t), classification_targets(y_p)
    if len(y_t) != len(y_p):
        raise ValueError("Targets and predictions must have equal lengths")
    if (y_t.dtype.kind in 'US') != (y_p.dtype.kind in 'US'):
        raise ValueError("Targets and predictions must use compatible label types")
    return y_t, y_p


def accuracy_score(y_t, y_p):
    y_t, y_p = paired_targets(y_t, y_p)
    return np.mean(y_t == y_p)

def class_labels(y_t, y_p, labels=None):
    y_t, y_p = paired_targets(y_t, y_p)
    if labels is not None:
        labels = classification_targets(labels)
        if (labels.dtype.kind in 'US') != (y_t.dtype.kind in 'US'):
            raise ValueError("labels must use the same type as the targets")
        if len(np.unique(labels)) != len(labels):
            raise ValueError("labels must be unique")
        return labels

    b_labs = list(y_t) + list(y_p)
    return np.unique(np.array(b_labs))

def confusion_matrix(y_t, y_p, labels=None):
    y_t, y_p = paired_targets(y_t, y_p)

    labels = class_labels(y_t, y_p, labels)

    ematrix = np.zeros((len(labels), len(labels)), dtype=int)

    for i in range(len(y_t)):
        true_v = y_t[i]
        pred_v = y_p[i]
        if true_v not in labels or pred_v not in labels:
            continue
        true_idx = np.where(labels == true_v)[0][0]
        pred_idx = np.where(labels == pred_v)[0][0]

        ematrix[true_idx, pred_idx] = ematrix[true_idx, pred_idx] + 1

    return ematrix

def avg_score(scores, supports, avg):
    scores = np.array(scores)
    supports = np.array(supports)

    if avg == "macro":
        return np.mean(scores)

    if avg == "weighted":
        total = np.sum(supports)

        if total == 0:
            return 0

        return np.sum(scores*supports) / total

    if avg is not None:
        raise ValueError("avg must be macro, weighted, or None")
    return scores

def precision_score(y_t, y_p, avg="macro", labels=None):
    labels = class_labels(y_t, y_p, labels)
    full_labels = np.unique(np.concatenate((targets(y_t), targets(y_p), labels)))
    cmatrix = confusion_matrix(y_t, y_p, labels=full_labels)

    scores=[]
    supports=[]

    for label in labels:
        i = int(np.flatnonzero(full_labels == label)[0])
        tposi = cmatrix[i,i]
        fposi = np.sum(cmatrix[:,i]) - tposi

        support = np.sum(cmatrix[i, :])

        if tposi + fposi == 0:
            precis = 0
        else:
            precis = tposi / (tposi+fposi)

        scores.append(precis)
        supports.append(support)
    return avg_score(scores, supports, avg)

def recall_score(y_t, y_p, avg="macro", labels=None):
    labels = class_labels(y_t, y_p, labels)
    full_labels = np.unique(np.concatenate((targets(y_t), targets(y_p), labels)))
    cmatrix = confusion_matrix(y_t, y_p, labels=full_labels)

    scores = []
    supports = []

    for label in labels:
        i = int(np.flatnonzero(full_labels == label)[0])
        tposi = cmatrix[i,i]
        fneg = np.sum(cmatrix[i,:]) - tposi
        support = np.sum(cmatrix[i,:])

        if tposi + fneg == 0:
            re = 0
        else:
            re = tposi / (tposi+fneg)
        scores.append(re)
        supports.append(support)

    return avg_score(scores, supports, avg)

def f1_score(y_t, y_p, avg="macro", labels=None):
    labels = class_labels(y_t, y_p, labels)
    full_labels = np.unique(np.concatenate((targets(y_t), targets(y_p), labels)))
    cmatrix = confusion_matrix(y_t, y_p, labels=full_labels)

    scores = []
    supports = []

    for label in labels:
        i = int(np.flatnonzero(full_labels == label)[0])
        tposi = cmatrix[i,i]
        fposi = np.sum(cmatrix[:,i]) - tposi
        fneg = np.sum(cmatrix[i,:]) - tposi
        support = np.sum(cmatrix[i,:])

        if tposi + fposi == 0:
            precis = 0
        else:
            precis = tposi / (tposi+fposi)

        if tposi + fneg ==0:
            re = 0
        else:
            re = tposi / (tposi+fneg)

        if precis + re == 0:
            f1 = 0
        else:
            f1 = 2*precis*re/(precis+re)

        scores.append(f1)
        supports.append(support)
    return avg_score(scores, supports, avg)
