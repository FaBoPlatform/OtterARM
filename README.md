# Otter ARM

## Otter ARMとは？

小型ロボットアームのプロジェクトです。TransfomerベースのAIロボットアーム制御を主軸においたロボットアームです。

![](./img/logo.png)

## 基本構成

Single

- 2カメラ
- Follower ARM x 1
- Leader ARM x 1

Double

- 4カメラ
- Follower ARM x 2
- Leader ARM x 2

## 対応する学習モデル

- ACT(Alohaの学習モデル)

## テレオペ

Single

```
python teleop.py --pair 1
```
Double

```
python teleop.py --pair 2
```

## データセットの作成

Single

```
python record.py --task test1 --pair 1
```
Double

```
python record.py --task test1 --pair 2
```

## 動画作成

直前のタスクの最新のエピソード

```
python movie.py --task test1
```

タスクの名前指定

```
python movie.py --task test1 --name episode_10.hdf5
```
