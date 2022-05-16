# ILLUSTPOSE　-OpenPoseFinetuning-
Openposeの既存モデルに対してファインチューニングを行い、イラストに対しての姿勢推定の精度を向上させたモデルを作成するためのリポジトリです。

## IllustPose概要
自動アニメーション生成器を作成するためにイラストの姿勢推定を行う必要であったが、既存の姿勢推定器であるOpenposeは現実世界の人間のデータセットを用いて作成されているためイラストのキャラクターでは精度が低くなってしまっている。そのため、イラストでオリジナルのデータセットを自作し、Openposeの既存モデルに対してFinetuningを行いました。その結果0.47から0.68に精度を向上させることに成功しました。

## データセット
Tagged Anime Illustrationsのキャラクター画像に対してCOCO keypoint challenge datasetをもとにアノテーション情報を手打ちで3300枚(train:2700,val:300,test300)作成しました。データセット作成用のプログラムは別のリポジトリ内に配置しております。マスクデータは50px四方のマスクを作成し、使用しております。dropboxからダウンロードし、dataフォルダとして活用してください。COCO keypoint challenge datasetは元サイトにてダウンロードをしてください。(https://cocodataset.org/#keypoints-2016)

## 結果
* 元画像(左)、ファインチューニング前(中央)、ファインチューニング後(右)のテスト比較画像
* 複雑なポーズのキャラクター
* 複雑な衣装のキャラクター
* 後ろ向きのキャラクター
* モノトーン調のキャラクター

## 参考文献
このリポジトリは小川雄太郎氏による「つくりながら学ぶ！PytTorchによる発展ディープラーニング」のOpenPoseを参考に作成を行いました。

Created with reference to OpenPose by YutaroOgawa/pytorch_advanced. Reference: https://github.com/YutaroOgawa/pytorch_advanced

元のイラストデータセット（Tagged Anime Illustrations）
https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations

COCO keypoint challenge dataset
https://cocodataset.org/#keypoints-2016
