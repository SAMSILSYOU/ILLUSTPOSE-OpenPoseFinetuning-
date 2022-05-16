# ILLUSTPOSE　-OpenPoseFinetuning-
Openposeの既存モデルに対してファインチューニングを行い、イラストに対しての姿勢推定の精度を向上させたモデルを作成するためのリポジトリです。

## IllustPose概要
自動アニメーション生成器を作成するためにイラストの姿勢推定を行う必要であったが、既存の姿勢推定器であるOpenposeは現実世界の人間のデータセットを用いて作成されているためイラストのキャラクターでは精度が低くなってしまっている。そのため、イラストでオリジナルのデータセットを自作し、Openposeの既存モデルに対してFinetuningを行いました。その結果0.47から0.68に精度を向上させることに成功しました。

## データセット
Tagged Anime Illustrationsのキャラクター画像に対してCOCO keypoint challenge datasetをもとにアノテーション情報を手打ちで3300枚(train:2700,val:300,test300)作成しました。
![7](https://user-images.githubusercontent.com/105159848/168655064-7550b055-6069-4ca7-b345-67be6c13c573.jpg)

データセット作成用のプログラムは別のリポジトリ内に配置しております。

マスクデータは50px四方のマスクを作成し、使用しております。

![MaskImg](https://user-images.githubusercontent.com/105159848/168655122-559f2bc0-1e06-4e8e-90a7-07a15c741879.jpg)

dropbox(<https://www.dropbox.com/s/mc1z9sl5vym06a1/data.zip?dl=0>)からダウンロードし、dataフォルダとして活用してください。COCO keypoint challenge datasetは元サイトにてダウンロードをしてください。(<https://cocodataset.org/#keypoints-2016>)

## 使い方
IllustPoseディレクトリ内でdataファイル(<https://www.dropbox.com/s/mc1z9sl5vym06a1/data.zip?dl=0>)とweightファイル(<https://www.dropbox.com/s/681x9lq3n3ia8rx/weights.zip?dl=0>)の解凍を行ってください。必要なライブラリを各自インポートしてください。

illustpose_train.pyを実行することで元モデルであるpose_model_scratch.pthに対してファインチューニングを行うことができます。

illustpose_test.pyを実行することで指定したモデルで任意の画像の姿勢推定の結果画像を表示させます。

## 結果
* 元画像(左)、ファインチューニング前(中央)、ファインチューニング後(右)のテスト比較画像
![1](https://user-images.githubusercontent.com/105159848/168654651-ff4eb521-a6b9-436b-86c0-1fe5f8aabeab.png)

* 複雑なポーズのキャラクター
![2](https://user-images.githubusercontent.com/105159848/168654778-2da34a7f-32c5-426b-b246-8d11497f0f37.png)

* 複雑な衣装のキャラクター
![3](https://user-images.githubusercontent.com/105159848/168654856-db1f13de-8093-44e5-835a-4150bd493ab4.png)

* 後ろ向きのキャラクター
![4](https://user-images.githubusercontent.com/105159848/168654876-167d3d22-aecb-4f12-b6b1-5230565d489f.png)

* モノトーン調のキャラクター
![5](https://user-images.githubusercontent.com/105159848/168654917-16697eb6-49ab-437c-b276-1257c733739d.png)

* 各パーツにおける元モデルとの精度比較
![6](https://user-images.githubusercontent.com/105159848/168655014-772bd771-4cc2-45eb-abbc-df0ed8b2ddbf.png)

## 参考文献
このリポジトリは小川雄太郎氏による「つくりながら学ぶ！PytTorchによる発展ディープラーニング」のOpenPoseを参考に作成を行いました。

Created with reference to OpenPose by YutaroOgawa/pytorch_advanced. Reference: https://github.com/YutaroOgawa/pytorch_advanced

元のイラストデータセット（Tagged Anime Illustrations: 
https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations

COCO keypoint challenge dataset: 
https://cocodataset.org/#keypoints-2016
