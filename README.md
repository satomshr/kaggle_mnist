# kaggle_mnist / script for Digit Recognizer with kaggle_mnist

- digit_recognizer_SVM1.ipynb
  - svm.SVC を使って，デフォルトのハイパーパラメータで推定
  - accuracy_score ; 0.9891428571428571
  - submitted score ; 0.97521
  - 1578 / 2597 = 0.76076

- digit_recognizer_SVM2.ipynb
  - smv.SVC を使って，ハイパーパラメータはグリッドサーチで最適化。
  - グリッドサーチ時には，1000 個のデータを使用 (3000 に増やしても，変化なし)
  - Best Model Parameter:  {'C': 10, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}
  - accuracy_score ; 0.9999047619047619
  - submitted score ; 0.98207
  - 1305 / 2598 = 0.5023

- digit_recognizer_SVM3.ipynb
  - smv.SVC を使って，ハイパーパラメータはグリッドサーチで最適化
  - グリッドサーチ時には，3000 個のデータを使用
  - C のサーチ範囲を狭くした ; C : [2, 5, 10, 20, 50] ← 前回は C=10 が最適だったので，その周囲を細かく探索
  - Best Model Parameter:  {'C': 5, 'decision_function_shape': 'ovo', 'kernel': 'rbf'} ← C が変わった
  - accuracy_score ; 0.9993095238095238
  - submitted score ; 0.98167 :-p

- digit_recognizer_SVM4.ipynb
  - smv.SVC を使って，ハイパーパラメータはグリッドサーチで最適化
  - グリッドサーチ時には，3000 個のデータを使用
  - C のサーチ範囲を range(5, 50)。それ以外は decision_function_shape="ovo"，kernel="rbf" で固定
  - Best Model Parameter:  {'C': 8, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}
  - accuracy_score ; 0.9998095238095238
  - submitted score ; 0.98217
  - 1301 / 2599

- digit_recognizer_MLP1.ipynb
  - MLPClassfier を使用。ハイパーパラメタはデフォルト
  - Kaggle の notebook を利用。私の PC より圧倒的に速い
  - accuracy_score ; 1.0
  - submitted score ; 0.97360

- digital_recognizer_MLP2.csv
  - MLPClassfier を使用。hidden_layer_sizes = (100, 100)
  - Kaggle の notebook を利用
  - accuracy_score ; 1.0
