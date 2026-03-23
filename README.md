# data 221 — assignment 4 (sabir sherefa)

## files

| file | what it does |
|------|----------------|
| `q1_dataset_exploration.py` | breast cancer data shapes, class names, counts |
| `q2_decision_tree_entropy.py` | decision tree with entropy, train/test accuracy |
| `q3_constrained_tree.py` | tree with `max_depth`, top 5 features |
| `q4_neural_network.py` | scaled inputs, neural net, train/test accuracy |
| `q5_model_comparison.py` | confusion matrices for tree vs neural net |
| `q6_cnn_fashion_mnist.py` | fashion-mnist cnn, test accuracy |
| `q7_cnn_error_analysis.py` | confusion matrix + misclassified examples |

## how to run

Use a python that has `tensorflow` and `scikit-learn` (your course venv).

From this folder:

```bash
python q1_dataset_exploration.py
python q2_decision_tree_entropy.py
python q3_constrained_tree.py
python q4_neural_network.py
python q5_model_comparison.py
python q6_cnn_fashion_mnist.py
python q7_cnn_error_analysis.py
```

`q5` and `q7` open plot windows (two confusion matrices in q5; one confusion matrix plus three wrong-image figures in q7). If there is no display, run:

```bash
export MPLBACKEND=Agg
```

then run those scripts (plots still get created; you may see a harmless warning about non-interactive figures).

## note on numbers

Exact accuracy can move a little with library versions and random seeds. Your code should still show sensible train/test gaps and cnn test accuracy usually lands near the high 0.8x–0.9x range on fashion mnist with this small network.
