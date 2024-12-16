# YOLOv9のファインチューニング

公式のリポジトリからフォークして、独自のデータセットでファインチューニングを行うためのリポジトリです。

Implementation of paper - [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>


## Performance 

MS COCO

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv9-T**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt) | 640 | **38.3%** | **53.1%** | **41.3%** | **2.0M** | **7.7G** |
| [**YOLOv9-S**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt) | 640 | **46.8%** | **63.4%** | **50.7%** | **7.1M** | **26.4G** |
| [**YOLOv9-M**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt) | 640 | **51.4%** | **68.1%** | **56.1%** | **20.0M** | **76.3G** |
| [**YOLOv9-C**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt) | 640 | **53.0%** | **70.2%** | **57.8%** | **25.3M** | **102.1G** |
| [**YOLOv9-E**](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt) | 640 | **55.6%** | **72.8%** | **60.6%** | **57.3M** | **189.0G** |
<!-- | [**YOLOv9 (ReLU)**]() | 640 | **51.9%** | **69.1%** | **56.5%** | **25.3M** | **102.1G** | -->

<!-- tiny, small, and medium models will be released after the paper be accepted and published. -->

## Useful Links

<details><summary> <b>Expand</b> </summary>

Custom training: https://github.com/WongKinYiu/yolov9/issues/30#issuecomment-1960955297
    
ONNX export: https://github.com/WongKinYiu/yolov9/issues/2#issuecomment-1960519506 https://github.com/WongKinYiu/yolov9/issues/40#issue-2150697688 https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461

ONNX export for segmentation: https://github.com/WongKinYiu/yolov9/issues/260#issue-2191162150

TensorRT inference: https://github.com/WongKinYiu/yolov9/issues/143#issuecomment-1975049660 https://github.com/WongKinYiu/yolov9/issues/34#issue-2150393690 https://github.com/WongKinYiu/yolov9/issues/79#issue-2153547004 https://github.com/WongKinYiu/yolov9/issues/143#issue-2164002309

QAT TensorRT: https://github.com/WongKinYiu/yolov9/issues/327#issue-2229284136 https://github.com/WongKinYiu/yolov9/issues/253#issue-2189520073

TensorRT inference for segmentation: https://github.com/WongKinYiu/yolov9/issues/446

TFLite: https://github.com/WongKinYiu/yolov9/issues/374#issuecomment-2065751706

OpenVINO: https://github.com/WongKinYiu/yolov9/issues/164#issue-2168540003

C# ONNX inference: https://github.com/WongKinYiu/yolov9/issues/95#issue-2155974619

C# OpenVINO inference: https://github.com/WongKinYiu/yolov9/issues/95#issuecomment-1968131244

OpenCV: https://github.com/WongKinYiu/yolov9/issues/113#issuecomment-1971327672

Hugging Face demo: https://github.com/WongKinYiu/yolov9/issues/45#issuecomment-1961496943

CoLab demo: https://github.com/WongKinYiu/yolov9/pull/18

ONNXSlim export: https://github.com/WongKinYiu/yolov9/pull/37

YOLOv9 ROS: https://github.com/WongKinYiu/yolov9/issues/144#issue-2164210644

YOLOv9 ROS TensorRT: https://github.com/WongKinYiu/yolov9/issues/145#issue-2164218595

YOLOv9 Julia: https://github.com/WongKinYiu/yolov9/issues/141#issuecomment-1973710107

YOLOv9 MLX: https://github.com/WongKinYiu/yolov9/issues/258#issue-2190586540

YOLOv9 StrongSORT with OSNet: https://github.com/WongKinYiu/yolov9/issues/299#issue-2212093340

YOLOv9 ByteTrack: https://github.com/WongKinYiu/yolov9/issues/78#issue-2153512879

YOLOv9 DeepSORT: https://github.com/WongKinYiu/yolov9/issues/98#issue-2156172319

YOLOv9 counting: https://github.com/WongKinYiu/yolov9/issues/84#issue-2153904804

YOLOv9 speed estimation: https://github.com/WongKinYiu/yolov9/issues/456

YOLOv9 face detection: https://github.com/WongKinYiu/yolov9/issues/121#issue-2160218766

YOLOv9 segmentation onnxruntime: https://github.com/WongKinYiu/yolov9/issues/151#issue-2165667350

Comet logging: https://github.com/WongKinYiu/yolov9/pull/110

MLflow logging: https://github.com/WongKinYiu/yolov9/pull/87

AnyLabeling tool: https://github.com/WongKinYiu/yolov9/issues/48#issue-2152139662

AX650N deploy: https://github.com/WongKinYiu/yolov9/issues/96#issue-2156115760

Conda environment: https://github.com/WongKinYiu/yolov9/pull/93

AutoDL docker environment: https://github.com/WongKinYiu/yolov9/issues/112#issue-2158203480

</details>

## 環境

- pyenv
- Python 3.10.11
- cuda 11.8

## Setup

### 1. リポジトリをクローン

```bash
git clone git@github.com:TechC-SugarCane/train-YOLOv9.git

cd train-YOLOv9
```

### 2. Pythonの環境構築

`pyenv`を使うので、パソコンに入っていない人は[CONTRIBUTING.md](https://github.com/TechC-SugarCane/.github/blob/main/CONTRIBUTING.md#pyenv-pyenv-win-%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)を参考にしながらインストールしてください。

```bash
pyenv install
```

### 3. 仮想環境を作成

```bash
python -m venv .venv
```

### 4. 仮想環境を有効化

```bash
# mac
source .venv/bin/activate

# windows
.venv\Scripts\activate
```

※ 環境から抜ける場合は、`deactivate`コマンドを実行してください。

### 5. 依存パッケージをインストール

```bash
# CPUで推論を行う場合
pip install -r requirements-cpu.txt

# GPUで推論を行う場合
pip install -r requirements-gpu.txt
```

## Training

事前学習済みモデルとして`yolov9-e.pt`を使用するので、[こちら](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt)からダウンロードして`weights`ディレクトリに配置してください。

データセットは 一階層上に[manage-datasetリポジトリ](https://github.com/TechC-SugarCane/manage-dataset)をcloneし、[manage-dataset/README](https://github.com/TechC-SugarCane/manage-dataset/blob/main/README.md)に従ってダウンロードしてください。

```shell
# clone済みの人はスキップ
cd ..
git clone git@github.com:TechC-SugarCane/manage-dataset.git
```

学習後の結果は`runs/train/<--name(番号)>`に保存されます。

学習でよいスコアが出た場合は、`runs/train/<--name(番号)>/`にREADME.mdを作成してください。
その際は、[こちら](./runs/train/README.md)を参考に作成してください。

``` shell
# Sugarcaneのファインチューニング
python train_dual.py --workers 8 --device 0 --batch 16 --data ../manage-dataset/datasets/sugarcane/data.yaml --img 640 --cfg models/detect/yolov9-s.yaml --weights './weights/yolov9-s.pt' --name yolov9-s-sugarcane --hyp hyp.scratch-sugarcane.yaml --epochs 300 --close-mosaic 15

# Pineappleのファインチューニング
python train_dual.py --workers 8 --device 0 --batch 16 --data ../manage-dataset/datasets/pineapple/data.yaml --img 640 --cfg models/detect/yolov9-s.yaml --weights './weights/yolov9-s.pt' --name yolov9-s-pineapple --hyp hyp.scratch-pineapple.yaml --epochs 300 --close-mosaic 15


# 可変オプション
# --name: 結果を保存するディレクトリ名
# --workers: CPUのコア数  2, 4, 8 etc...
# --batch: バッチサイズ  8, 16, 32, 64 etc...
# --epochs: 学習回数  100～500?
# --close-mosaic: close mosaic(data augmentation)の確率  0～100

# ↓ 追加していいかも？
# --patience: EarlyStoppingのパラメータ  100(default), 20, 25 etc...
# --optimizer: 最適化アルゴリズム  SDG(default), Adam, AdamW
# --seed: シード値  0(default), 1, 2 etc...
# --cos-lr, --flat-cos-lr, --fixed-lr: スケジューラー どれか一つ
```

ハイパーパラメーターは自由に調整してください。`data/hyps/`にあります。

- サトウキビ: `hyp.scratch-sugarcane.yaml`
- パイナップル: `hyp.scratch-pineapple.yaml`

## Export

学習したモデルをONNX形式に変換する場合は、以下のコマンドを実行してください。

通常の`export.py`を使うとうまく変換されないので、ultralyticsを使ったやり方で行います。

```shell
# --weightsのパスを変更してください
# --dataも必要に応じて変更してください
python export_ultralytics.py --data ../manage-dataset/datasets/sugarcane/data.yaml --weights ./runs/train/<--name(番号)>/weights/best.pt --device 0 --format onnx --imgsz 640
```

上記の場合、`./runs/train/<--name(番号)>/weights/best.onnx`に変換されたモデルが保存されます。

## コントリビューター向けガイドライン

コントリビューター向けのガイドラインについては、こちらの[CONTRIBUTING.md](https://github.com/TechC-SugarCane/.github/blob/main/CONTRIBUTING.md)を参照してください。

### ※ 注意

このリポジトリはforkなので、Pull Requestを送る際はこのリポジトリに対して送るようにしてください。

デフォルトだとbaseリポジトリが公式のリポジトリになっているので、注意してください。

`Comparing changes`でのドロップダウン(`base repository`)を、`TechC-SugarCane/train-YOLOv9`に変更してください。画面が遷移したら大丈夫です。
