# 設計書: PINNによる1次元熱方程式 & 将来拡張Navier–Stokes対応

## 1. プロジェクト概要
本プロジェクトは、Physics-Informed Neural Networks (PINN) を用いて、まず1次元熱方程式を解き、将来的にNavier–Stokes方程式へ拡張することを目的とする。  
コードはモジュール化され、将来的な方程式差し替えや追加に大幅な構造変更を必要としない設計とする。

---

## 2. プロジェクトフォルダ構成
```
pinn_project/
│
├── configs/                      # 設定ファイル群
│   ├── train_settings.py          # ネットワーク構造・学習パラメータ設定
│   ├── physics_params_heat.py     # 1次元熱方程式の物理パラメータ
│   └── physics_params_ns.py       # Navier–Stokes用パラメータ（将来対応）
│
├── data/                         # データ格納
│   ├── synthetic/                 # 合成データ（解析解・CFD解）
│   ├── raw/                       # 実験や観測データ
│   └── processed/                  # 前処理後データ
│
├── src/                          # 実装コード
│   ├── models/                    # ニューラルネットワーク構造
│   │   ├── pinn_heat.py           # 1次元熱方程式用PINNモデル
│   │   ├── pinn_ns.py             # Navier–Stokes用PINNモデル（将来）
│   │   └── base_pinn.py           # 共通モデルクラス
│   │
│   ├── physics/                   # PDEや残差計算
│   │   ├── heat_equation.py       # 1次元熱方程式のPDE定義
│   │   ├── navier_stokes.py       # Navier–Stokes方程式（将来）
│   │   └── loss_functions.py      # 残差損失・データ損失計算
│   │
│   ├── dataio/                    # データ入出力
│   │   ├── data_loader.py         # CSV/HDFからデータ読み込み
│   │   ├── data_preprocessor.py   # 前処理（正規化など）
│   │   └── dataset_generator.py   # 合成データ生成（解析解ベース）
│   │
│   ├── training/                  # 学習関連
│   │   ├── trainer.py              # 学習ループ（Adam→LBFGS）
│   │   ├── scheduler.py            # 学習率スケジューラ（必要なら）
│   │   └── seed_utils.py           # Seed固定処理
│   │
│   ├── evaluation/                # 評価・可視化
│   │   ├── evaluator.py            # RMSE評価など
│   │   ├── plot_results.py         # 結果の可視化
│   │   └── metrics.py              # 評価指標計算
│   │
│   └── utils/                      # 共通ユーティリティ
│       ├── logger.py               # ロギング処理
│       ├── device_utils.py         # GPU/CPU切替
│       └── file_utils.py           # ファイル操作
│
├── tests/                         # テストコード
│   ├── test_models.py
│   ├── test_dataio.py
│   └── test_training.py
│
├── scripts/                       # 実行用スクリプト
│   ├── train_heat.py               # 熱方程式用学習実行
│   ├── train_ns.py                 # Navier–Stokes用（将来）
│   └── evaluate.py                 # 評価実行
│
├── requirements.txt               # Python依存パッケージ
├── README.md
└── LICENSE
```


---

## 3. 主なモジュール設計

### 3.1 configs/
- YAML形式で学習条件やネットワーク構造、損失重みを管理。
- PDE種類に応じて設定ファイルを切り替え可能。

### 3.2 data/
- 合成データ（解析解やCFD出力）と実データを分離。
- `processed/`には前処理済みテンソル形式のデータを保存。

### 3.3 src/physics/
- `pde_base.py`: PDE定義の共通インターフェース。
- `heat1d.py`: 1次元熱方程式の残差計算。
- `navier_stokes.py`: Navier–Stokes用の残差計算（将来追加）。

### 3.4 src/models/
- `pinn_base.py`: PINNの共通構造（MLPなど）。
- PDEごとにサブクラスを用意 (`pinn_heat1d.py` など)。

### 3.5 src/utils/
- データ読み込み・可視化・ログ出力・乱数Seed固定・設定読み込み。

### 3.6 src/training/
- `trainer.py`: 学習ループ実装。
- `losses.py`: PDE残差・観測データRMSEなどの計算。
- `optimizer_switch.py`: AdamからLBFGSへ切り替え。

### 3.7 src/evaluation/
- `evaluator.py`: 学習済みモデル評価。
- `metrics.py`: RMSEなどの評価指標。

---

## 4. 学習フロー（1次元熱方程式）

1. 設定ファイル読み込み（`config_util.py`）
2. データ読み込み & 前処理（`data_loader.py`）
3. モデル構築（`pinn_heat1d.py`）
4. 損失計算  
   - 物理残差: PDEを満たすか  
   - 観測一致: 実データとのRMSE
5. 最適化（Adam → LBFGS）
6. 評価（RMSE, 可視化）

---

## 5. Navier–Stokes対応方法
- PDE定義 (`navier_stokes.py`) とモデル定義 (`pinn_ns.py`) を追加。
- 設定ファイルをNavier–Stokes用に切り替え。
- `trainer.py` や `losses.py` は共通利用可能。

---

## 6. 実装計画（フェーズ分割）
1. **フェーズ1**: Heat1D実装（解析解ベースで検証）
2. **フェーズ2**: Heat1D + 実観測データ対応
3. **フェーズ3**: Navier–Stokes対応モジュール追加
4. **フェーズ4**: GPU最適化、マルチGPU化

---
