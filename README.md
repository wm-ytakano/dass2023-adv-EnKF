# 1次元線形移流拡散モデル + EnKFのPyTorch実装

データ同化夏の学校2023 課題2

## メモ

真値の強制項に加える乱数は、時間方向の相関を考慮するためAR(1)過程

$$ q_t = 0.8 q_{t-1} + 0.2 \mathcal{N}(0,1)$$

としている。サンプルコードではこの乱数は含まれていない。教科書5.3節(143ページ)の設定では「外力は15ステップ毎に入手するデータとし、ステップ間では線形内挿でを行って与える」とあり、15ステップ程度の時間スケールで乱数成分が時間変動する。