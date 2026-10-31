"""Render aggregate confusion matrices and fold scores from a recorded benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(summary_path, destination):
    result = json.loads(Path(summary_path).read_text())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    labels = result['labels']
    comparison = result['comparison']
    entries = [('Recording split, fixed k=5', comparison['recording']),
               ('Held-out actors, fixed k=5', comparison['actor']),
               ('Nested actor CV, pooled predictions', result['nested_actor_cv']['pooled'])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    for ax, (title, record) in zip(axes, entries):
        cm = np.asarray(record['confusion_matrix'])
        if cm.shape != (len(labels), len(labels)) or cm.sum() == 0:
            raise ValueError('Invalid recorded confusion matrix')
        normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True)>0,
                               out=np.zeros_like(cm, dtype=float))
        ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        for row, col in np.ndindex(cm.shape):
            ax.text(col, row, str(cm[row, col]), ha='center', va='center', fontsize=9,
                    color='white' if normalized[row, col] > .55 else '#182b3a')
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
               xticklabels=labels, yticklabels=labels, xlabel='Predicted expression', ylabel='Recorded label',
               title=f"{title}\nAccuracy {record['accuracy']:.1%}, macro-F1 {record['macro_f1']:.3f}")
        ax.tick_params(axis='x', labelrotation=55)
    fig.suptitle('RAVDESS speech feature-cache experiments | counts shown, shading normalized by row', fontsize=14)
    fig.savefig(destination/'confusion-matrices.png', dpi=170, bbox_inches='tight', pad_inches=.2, metadata={'Software': 'Matplotlib'})
    plt.close(fig)
    folds = result['nested_actor_cv']['outer_folds']
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    x = np.arange(len(folds)); width = .35
    ax.bar(x-width/2, [r['accuracy'] for r in folds], width, label='Accuracy', color='#227c9d')
    ax.bar(x+width/2, [r['macro_f1'] for r in folds], width, label='Macro-F1', color='#e59f27')
    ax.set(ylim=(0, 1), xticks=x,
           xticklabels=[f"Fold {r['fold']}\nk={r['k']}" for r in folds], ylabel='Score',
           title='Nested actor-grouped CV: four held-out actors per outer fold')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(destination/'actor-cv-folds.png', dpi=170, bbox_inches='tight', pad_inches=.2, metadata={'Software': 'Matplotlib'})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.summary, args.output_dir)
