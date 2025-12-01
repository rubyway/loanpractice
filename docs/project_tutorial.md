# Loan Prediction 项目全链路教程（ML + DL 实战）

> 目标：以本仓库为例，从数据准备、特征工程、模型构建、调参思路到评测与提交，系统梳理“贷款逾期预测”这一经典任务的 ML/DL 实践路径，帮助你由浅入深掌握真实项目流程。

---

## 1. 项目概览

| 阶段 | 关键内容 | 对应文件 |
| --- | --- | --- |
| 数据准备 | `train.csv`、`test.csv`、`sample_submission.csv` | `data/` |
| 预处理与特征工程 | 缺失值填补、LabelEncoder、StandardScaler | `src/data_preprocessing.py` |
| 传统 ML 模型 | Logistic、RandomForest、GradientBoosting、XGBoost、LightGBM、CatBoost | `src/models.py` |
| 深度学习模型 | FeedForward NN、Transformer NN（PyTorch） | `src/deep_learning.py` |
| Ensemble 策略 | Voting / Stacking / Weighted | `src/ensemble.py` |
| 主训练脚本 | 统一训练、多模型对比、生成预测 | `main.py` |
| 快速实验 & 提交 | `train.py`（简版）、`run_experiments.py`（批量实验）、`kaggle_best_model.ipynb`（提交版） | 根目录 |

---

## 2. 数据与预处理

### 2.1 数据结构

每条样本由借款人信息 + 贷款信息组成，核心字段如下：

- **连续特征**：`annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`
- **类别特征**：`gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`
- **目标**：`loan_paid_back`（1 = 准时还款，0 = 违约）
- **ID**：`id`

### 2.2 预处理流程（`preprocess_data`）

1. **缺失值**：数值列填中位数、类别列填众数（无则填 `Unknown`）
2. **类别编码**：`LabelEncoder` 在 train+test 合并后拟合，避免测试集出现新类别
3. **数值标准化**：`StandardScaler` 先将目标列从 DataFrame 复制出来，数值列转换成 `float64` 再缩放，确保类型兼容
4. **列对齐**：只保留 train/test 同时存在的特征列（`test_feature_cols`）
5. **返回**：`X_train`、`y_train`（int）、`X_test`、`test_ids`、`feature_cols`、`encoders`、`scaler`

Tip：Notebook 版本 (`kaggle_best_model.ipynb`) 复用了上述逻辑，可在 Kaggle 环境中直接运行。

---

## 3. 传统机器学习模型

关键文件：`src/models.py`

### 3.1 基础模型

- Logistic Regression（`max_iter=1000`，可调 `C`、`solver`）
- RandomForest（多样本随机子集 + 决策树集合）
- GradientBoosting（串行弱 Learner 迭代优化）
- KNN / SVM（可选）

### 3.2 Boosting 系列

- **XGBoost**：`n_estimators=200`、`learning_rate=0.05`、`max_depth=7`
  - 关键参数：`min_child_weight`、`subsample`、`colsample_bytree`、`eval_metric`
- **LightGBM**：`num_leaves`、`subsample`、`colsample` 等
- **CatBoost**：擅长处理类别特征（本项目中暂以数值编码为主）

### 3.3 调参思路

1. 从基础参数开始（例如 `n_estimators=100`）
2. 先调 **模型复杂度**（`max_depth`、`num_leaves`），后调 **学习率** 与 **数据采样** 参数
3. 使用 `train_and_evaluate_all` 自动化跑所有模型并记录指标（见 `main.py`）

---

## 4. 深度学习模型（PyTorch）

关键文件：`src/deep_learning.py`

### 4.1 FeedForwardNN

- 结构：线性层 → BatchNorm → ReLU → Dropout，支持自定义隐藏层结构
- 默认：`(128, 64, 32)`，`dropout=0.3`
- `train_nn_model`：支持 early stopping、ReduceLROnPlateau Scheduler、GPU 自动检测

### 4.2 TransformerNN（表格轻量版）

- 每个特征视为一个 token，包含：
  - 可学习的 `[CLS]` token 和位置编码
  - 自定义 Transformer block（支持双 Linear 或 gated 三 Linear FFN）
  - Head：GELU + Dropout → 单输出
- 参数：`d_model=64`、`nhead=4`、`num_layers=2`、`ff_hidden=128`
- 用途：探索更复杂交互关系

### 4.3 提示

- 对 tabular 数据，MLP/GBDT 通常 baseline 更强；引入 Transformer 可探索非线性模式，但请关注训练成本
- `main.py --use-nn` 可同时训练多个 NN 变体

---

## 5. Ensemble 方法

关键文件：`src/ensemble.py`

| 方法 | 原理 | 场景 |
| --- | --- | --- |
| Voting | 多模型投票或概率平均 | 所有模型性能相近时 |
| Stacking | 元学习器对 base 模型预测进行再训练 | 追求更强泛化 |
| Weighted Average | 对各模型概率设权重 | 需要精细调融合时 |

`main.py --ensemble {voting,stacking,weighted,none}` 控制最终预测方式。“none” 会直接选 `get_best_model`（默认按 F1）。Weighted 模式会通过 `optimize_ensemble_weights` 搜索最优组合。

---

## 6. 运行脚本与流程

### 6.1 主脚本 `main.py`

```bash
uv run python main.py \
  --train data/train.csv \
  --test data/test.csv \
  --use-nn \
  --nn-models feedforward transformer \
  --ffn-variant gated \
  --ensemble stacking \
  --output predictions.csv
```

功能：

1. 加载 + 预处理 + train/val 划分
2. 训练 ML 模型（可选 NN）
3. 输出对比表（Accuracy、Precision、Recall、F1、ROC-AUC）
4. 按选定 Ensemble 生成测试集预测 → `predictions.csv`，包含 `loan_paid_back` 与 `probability`

### 6.2 快速脚本 `train.py`

- 结构简化，培训几个核心模型，适合快速 baseline
- 输出 `predictions.csv`（未含概率）

### 6.3 批量实验 `run_experiments.py`

- 预设多种组合（Voting、Stacking、NN、Best Single、Transformer）
- 自动算 ROC-AUC、F1 等指标，并将每次实验写入 `logs/experiment_summary.csv`
- 便于快速对比方案、寻找最佳配置（当前最优：Gradient Boosting ROC-AUC ≈ 0.9187）

### 6.4 运行记录 `log_runs.py`

- 单次运行所有传统模型 +（可选）NN，将结果记录至 CSV（含 run_id、config、metrics）
- 方便日志式跟踪、结果复现

---

## 7. Kaggle 提交流程（`kaggle_best_model.ipynb`）

1. **数据读取**：支持 Kaggle `/kaggle/input` 与本地 `data/`
2. **预处理**：与项目一致
3. **候选模型**：根据 `run_experiments.py` 选出的最优 Gradient Boosting
4. **交叉验证**：StratifiedKFold 计算 ROC-AUC、RMSE
5. **全量训练**：用最佳配置训练整个训练集
6. **预测与导出**：生成 `submission.csv`，包含 `id`, `loan_paid_back`, `probability`

> 在 Kaggle Notebook 依次运行所有单元即可直接提交；Notebook 中的注释/内容为中英文双语。

---

## 8. 实战关键点总结

1. **数据一致性**：train/test 统一预处理、列对齐，否则模型会崩
2. **类型管理**：pandas 未来将强制 dtype 兼容，务必在缩放前将数值列转 `float64`
3. **LabelEncoding**：train+test 拼接拟合，避免测试集出现 unseen category
4. **模型多样性**：传统 ML（尤其 Gradient Boosting/XGBoost）在 tabular 数据上表现依然强劲
5. **深度学习探索**：PyTorch 提供灵活扩展空间（FeedForward/Transformer），可结合 `--nn-models` 调参
6. **Ensemble**：投票/堆叠/加权可提升表现，但要平衡训练成本
7. **指标选择**：本项目最终以 ROC-AUC 作为选模依据，更关注排序性能；其他场景可按 F1、Accuracy 等调整
8. **实验记录**：`run_experiments.py` + `log_runs.py` 帮你保持实验可追溯、可比较
9. **Kaggle 提交**：精简 Notebook 只保留最佳配置，减少不必要单元和运行时间，确保结果可复现

---

## 9. 示例学习路线

1. **入门**：运行 `train.py`，理解预处理和几个经典模型
2. **进阶**：阅读 `src/models.py`，学习多模型训练与 `train_and_evaluate_all`
3. **深入**：掌握 `src/deep_learning.py` 中 NN 架构，实现自定义模型
4. **集成与调参**：使用 `main.py` + `run_experiments.py` 对 hyperparameters、ensemble、NN 组合进行探索
5. **落地**：根据 `kaggle_best_model.ipynb` 结构，打造自己的 Kaggle 提交流程

---

通过本教程，你可以从零到一搭建完整的贷款预测系统，掌握：

- 数据预处理与特征工程技巧
- 传统机器学习模型的选型与调参
- 深度学习模型在 tabular 任务中的适配
- Ensemble 融合策略
- 实验记录与日志化思路
- Kaggle 提交流程与落地实现

祝你在实际项目与 Kaggle 比赛中取得佳绩！
