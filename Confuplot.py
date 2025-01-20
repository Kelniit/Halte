import seaborn as sns

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def ConfusionMatrixHelplot(true_label, logits, classes):
  """
  Confusion Matrix Helplot

  >>> classes = [0, 1]
  >>> logits = np.round(model.predict(...))
  >>> ConfusionMatrixHelplot(test_label, logits, classes)
  """
  result = confusion_matrix(true_label, logits)
  plotresult = sns.heatmap(result, annot=True, fmt="g", cmap="Pastel1")
  plotresult.set_xticklabels(classes)
  plotresult.set_yticklabels(classes)
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.title('Confusion Matrix Plot');
