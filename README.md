# kaggle_mnist / script for Digit Recognizer with kaggle_mnist

- digit_recognizer_SVM1.ipynb
  - svm.SVC を使って，デフォルトのハイパーパラメータで推定
  - accuracy_score ; 0.9891428571428571
  - submitted score ; 0.97521
  - 1578 / 2597 = 0.76076

- digit_recognizer_SVM2.ipynb
  - smv.SVC を使って，ハイパーパラメータはグリッドサーチで最適化。
  - グリッドサーチ時には，1000 個のデータを使用
  - Best Model Parameter:  {'C': 10, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}
  - accuracy_score ; 0.9999047619047619
  - submitted score ; 0.98207
  - 1305 / 2598 = 0.5023
