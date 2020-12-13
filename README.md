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

- digit_recognizer_SVM5a.csv
  - データの加工
    - すべてが 0 の列を drop
    - 255 で割る
    - 行ごとに平均 (mean) と標準偏差 (std) を計算し，列に追加
  - SVM1
    - 3000 個のデータを使って GridSearchCV
    - Best Model Parameter: {'C': 10, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}
    - Train score: 1.0
    - Cross Varidation score: 0.9396666666666667
    - Total Train score: 0.9546190476190476
  - SVM2
    - 全データを使って GridSearchCV。C : [2, 5, 10, 20, 50]
    - Best model parameter :　{'C': 10, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}
    - Train score: 0.9999523809523809
    - Cross Varidation score: 0.9806190476190476
  - SVM3
    - 全データを使って GridSearchCV。C : [6, 8, 10, 13, 16, 19]
    - Best model parameter :　{'C': 8, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}
    - Train score: 0.9999047619047619
    - Cross Varidation score: 0.9807857142857144
  - SVM4
    - 全データを使って GridSearchCV。C : [7, 8, 9]
    - Best Model Parameter: {'C': 8, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}
    - Train score: 0.9999047619047619
    - Cross Varidation score: 0.9807857142857144
  - SVM5
    - 全データを使って, 最適パラメータで学習して，予測
    - accuracy_score : 0.9999047619047619
    - submitted score : 0.98228
    - 1398 / 2768
    - gamma も最適化したほうが良さそうだ
  - Kaggle notebook Ver.5 (Ver.4 も同じだと思うが，save を途中で止めた)
  - ハイパーパラメータは変わってないが，スコアは上がった。drop，mean, std が効いたか?

- digit_recognizer_SVM6a.csv
  - SMV1
    - 3000 個のデータを使って GridSearchCV。C と gamma を振る
    - Best model parameter : {'C': 10, 'gamma': 0.01}
    - Train score: 1.0
    - Cross Varidation score: 0.9343333333333332 (上より悪い)
    - Total Train score: 0.9514523809523809 (上より悪い)
    - CV = 0.9343 となる組み合わせは…
      - 0.9343 (+/-0.0104) for {'C': 10, 'gamma': 0.01}
      - 0.9343 (+/-0.0104) for {'C': 100, 'gamma': 0.01}
      - 0.9343 (+/-0.0104) for {'C': 1000, 'gamma': 0.01}
      - 0.9343 (+/-0.0104) for {'C': 10000, 'gamma': 0.01}
      - 0.9343 (+/-0.0104) for {'C': 100000, 'gamma': 0.01}
      - 0.9037 (+/-0.0195) for {'C': 100, 'gamma': 0.001}
      - 0.9017 (+/-0.0194) for {'C': 1000, 'gamma': 0.001}
      - 0.9017 (+/-0.0194) for {'C': 10000, 'gamma': 0.001}
      - 0.9017 (+/-0.0194) for {'C': 100000, 'gamma': 0.001}
  - SVM2
    - 3000 個のデータを使って GridSearchCV。C と gamma を振る
    - Best model parameter : {'C': 3, 'gamma': 0.03}
    - Train score: 1.0
    - Cross Varidation score: 0.9446666666666665 (まだ良いとは言えない)
    - Total Train score: 0.9583095238095238 (まだ良いとは言えない)
    - CV = 0.9446 となる組み合わせは…
      - 0.9447 (+/-0.0185) for {'C': 3, 'gamma': 0.03}
      - 0.9447 (+/-0.0185) for {'C': 10, 'gamma': 0.03}
      - 0.9447 (+/-0.0185) for {'C': 30, 'gamma': 0.03}
      - 0.9447 (+/-0.0185) for {'C': 100, 'gamma': 0.03}
      - 0.9447 (+/-0.0185) for {'C': 3000, 'gamma': 0.03}
      - 0.9447 (+/-0.0185) for {'C': 10000, 'gamma': 0.03}
      - 0.9447 (+/-0.0185) for {'C': 30000, 'gamma': 0.03}
      - 0.9447 (+/-0.0185) for {'C': 100000, 'gamma': 0.03}
  - SVM3
    - 5000 個のデータを使って GridSearchCV。C と gamma を振る。感じ的には，C を振っても変わらず，gamma の影響が大きい。今回の目的は，gamma の値を絞りに行くこと
    - Best Model Parameter: {'C': 10, 'gamma': 0.03}
    - Train score: 1.0
    - Cross Varidation score: 0.959
    - Total Train score: 0.9659285714285715
    - CV = 0.959 となる組み合わせは…
      - 0.9590 (+/-0.0126) for {'C': 10, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 30, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 100, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 3000, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 10000, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 30000, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 100000, 'gamma': 0.03}
  - SVM4
    - 5000 個のデータを使って GridSearchCV。C と gamma を振る。gamma の範囲を，より絞り込んだ。
    - Best model parameter : {'C': 10, 'gamma': 0.03}
    - Train score: 1.0
    - Cross Varidation score: 0.959
    - Total Train score: 0.9659285714285715
    - CV = 0.959 となる組み合わせは
      - 0.9590 (+/-0.0126) for {'C': 10, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 30, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 100, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 3000, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 10000, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 30000, 'gamma': 0.03}
      - 0.9590 (+/-0.0126) for {'C': 100000, 'gamma': 0.03}
  - SVM5
    - 8000 個のデータを使って GridSearchCV。gamma = 0.03 に固定して，C を振る
    - Best model parameter : {'C': 10, 'gamma': 0.03}
    - Train score: 1.0
    - Cross Varidation score: 0.9684999999999999
    - Total Train score: 0.973547619047619
    - CV=0.9685 になる組み合わせは
      - 0.9684 (+/-0.0079) for {'C': 3, 'gamma': 0.03}
      - 0.9685 (+/-0.0087) for {'C': 10, 'gamma': 0.03}
      - 0.9685 (+/-0.0087) for {'C': 30, 'gamma': 0.03}
      - 0.9685 (+/-0.0087) for {'C': 100, 'gamma': 0.03}
      - 0.9685 (+/-0.0087) for {'C': 3000, 'gamma': 0.03}
      - 0.9685 (+/-0.0087) for {'C': 10000, 'gamma': 0.03}
      - 0.9685 (+/-0.0087) for {'C': 30000, 'gamma': 0.03}
      - 0.9685 (+/-0.0087) for {'C': 100000, 'gamma': 0.03}
  - 最後
    - C=10, gamma=0.03 で全データで学習
    - accuracy_score : 1.0
    - submitted score : 0.98353 (1388/2803) ← 上がった


- digit_recognizer_MLP1.ipynb
  - MLPClassfier を使用。ハイパーパラメタはデフォルト
  - Kaggle の notebook を利用。私の PC より圧倒的に速い
  - accuracy_score ; 1.0
  - submitted score ; 0.97360

- digit_recognizer_MLP2.csv
  - MLPClassfier を使用。hidden_layer_sizes = (100, 100)
  - Kaggle の notebook を利用
  - accuracy_score ; 1.0
  - submitted score ; 0.97639

- digit_recognizer_MLP3a.csv
  1. 全データで 0 のピクセルを drop した
  2. GridSearchCV で，データ数 5000 個で粗くサーチ (cv=4)
    - Best Model Parameter: {'batch_size': 50, 'early_stopping': False, 'hidden_layer_sizes': 100, 'learning_rate_init': 0.01}
    - Train score: 0.998
    - Cross Varidation score: 0.9451999999999999
  3. GridSearchCV で，全データを使ってサーチ (cv=4)
    - Best Model Parameter: {'batch_size': 100, 'early_stopping': False, 'hidden_layer_sizes': 100, 'learning_rate_init': 0.001}
    - Train score: 1.0
    - Cross Varidation score: 0.9706190476190476
  4. 上記の結果で，全データを使って改めて学習
  5. 結果
    - submitted score ; 0.97510


  | Ver. | accuracy_score | submitted score |
  | ---- | ---- | ---- |
  | SVM1 | 0.9891428571428571 | 0.97521 |
  | SVM2 | 0.9999047619047619 | 0.98207 |
  | SVM3 | 0.9993095238095238 | 0.98167 |
  | SVM4 | 0.9998095238095238 | 0.98217 |
  | SVM5a | 0.9999047619047619 | 0.98228 |
  | SVM6a | 1.0 | 0.98353 |
  | MLP1 | 1.0 | 0.97360 |
  | MLP2 | 1.0 | 0.97639 |
  | MLP3 | 1.0 | 0.97510 |

- create_new_features1.ipynb
  - 新しい特徴量を探索するためのスクリプト. このスクリプトは, 画像を表示させて, 数字毎の画像の平均値を計算する
  - 2 分割，4 分割, 9 分割した領域の平均と分散などの特徴量が考えられる
  - 数字毎に平均した画像との相関をとっても良いのでは? という気もしている

- image_mean.jpg
  - 数字毎の画素の平均値の画像
