# Loan Prediction Machine Learning Project / 贷款预测机器学习项目

This project implements multiple machine learning and deep learning models to predict loan status based on borrower information.

本项目实现了多种机器学习和深度学习模型，用于根据借款人信息预测贷款状态。

## Features / 功能特性

- **Multiple ML Models / 多种机器学习模型**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost / 逻辑回归、随机森林、梯度提升、XGBoost、LightGBM、CatBoost
- **Deep Learning / 深度学习**: Neural network models using PyTorch / 使用 PyTorch 的神经网络模型
- **Configurable NN Architectures / 可配置神经网络架构**: Feed-forward and transformer-based PyTorch models with selectable FFN variants / 支持前馈与 Transformer PyTorch 模型，并可选择不同 FFN 结构
- **Ensemble Methods / 集成方法**: Voting, Stacking, and Weighted Average ensembles / 投票、堆叠和加权平均集成
- **Automated Preprocessing / 自动预处理**: Handles missing values, categorical encoding, and feature scaling / 处理缺失值、分类编码和特征缩放

## Data Format / 数据格式

The project expects CSV files with the following columns: / 项目需要包含以下列的 CSV 文件：
- `annual_income`: Borrower annual income / 借款人年收入
- `debt_to_income_ratio`: Debt-to-income ratio / 负债收入比
- `credit_score`: Internal or bureau credit score / 信用评分
- `loan_amount`: Original loan amount / 贷款金额
- `interest_rate`: Loan interest rate / 贷款利率
- `gender`: Gender feature / 性别特征
- `marital_status`: Marital status / 婚姻状况
- `education_level`: Education level / 教育程度
- `employment_status`: Employment status / 就业状态
- `loan_purpose`: Declared purpose of the loan / 贷款目的
- `grade_subgrade`: Loan grade-subgrade string / 贷款等级-小等级
- `loan_paid_back`: Target (1 = fully paid, 0 = default) / 目标变量（1=已还清，0=违约）

## Installation / 安装

```bash
# Install uv if you have not already / 若尚未安装 uv，请先执行
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install dependencies / 创建虚拟环境并安装依赖
uv sync
```

## Usage / 使用方法

### Quick Training / 快速训练

```bash
uv run python train.py data/train.csv data/test.csv predictions.csv
```

### Full Training with Options / 带选项的完整训练

```bash
uv run python main.py --train data/train.csv --test data/test.csv --output predictions.csv
```

#### Options / 选项:
- `--train`: Path to training CSV (required) / 训练 CSV 文件路径（必需）
- `--test`: Path to test CSV (required) / 测试 CSV 文件路径（必需）
- `--original`: Path to original loan data CSV (optional) / 原始贷款数据 CSV 文件路径（可选）
- `--output`: Output file for predictions (default: predictions.csv) / 预测输出文件（默认：predictions.csv）
- `--use-nn`: Include neural network models / 包含神经网络模型
- `--nn-models`: Choose NN architectures (`feedforward`, `transformer`) / 选择神经网络架构（`feedforward`、`transformer`）
- `--ffn-variant`: Transformer FFN style (`double`, `gated`) / 设定 Transformer FFN 结构（`double`、`gated`）
- `--ensemble`: Ensemble method - voting, stacking, weighted, or none (default: voting) / 集成方法 - voting、stacking、weighted 或 none（默认：voting）
- `--val-size`: Validation set size ratio (default: 0.2) / 验证集大小比例（默认：0.2）

### Examples / 示例

```bash
# Use voting ensemble (default) / 使用投票集成（默认）
uv run python main.py --train data/train.csv --test data/test.csv --ensemble voting

# Use stacking ensemble / 使用堆叠集成
uv run python main.py --train data/train.csv --test data/test.csv --ensemble stacking

# Include neural network / 包含神经网络
uv run python main.py --train data/train.csv --test data/test.csv --use-nn

# Use best single model / 使用最佳单一模型
uv run python main.py --train data/train.csv --test data/test.csv --ensemble none

# Select transformer NN with gated FFN / 使用带 gated FFN 的 Transformer NN
uv run python main.py --train data/train.csv --test data/test.csv \
	--use-nn --nn-models transformer --ffn-variant gated
```

## Neural Network Options / 神经网络选项

- **Architectures / 架构**: Use `--nn-models feedforward transformer` to train both classic MLP and transformer-based networks in a single run. / 使用 `--nn-models feedforward transformer` 可在一次运行中同时训练传统 MLP 与 Transformer 网络。
- **FFN Variants / FFN 变体**: Transformer blocks support `--ffn-variant double` (传统双线性 FFN) 或 `--ffn-variant gated`（三线性门控 FFN，可比较表现）。
- **Device & Training / 设备与训练**: PyTorch 会自动检测 GPU/CPU，并在训练后将模型迁回 CPU。/ PyTorch 自动检测 GPU/CPU，训练完成后会把模型迁回 CPU。

## Run Logging / 运行记录

Use `log_runs.py` to capture every model configuration and validation metric into a CSV log for later analysis. / 使用 `log_runs.py` 将每个模型配置与验证指标写入 CSV 便于后续分析。

```bash
uv run python log_runs.py --train data/train.csv --test data/test.csv \
	--use-nn --nn-models feedforward transformer --ffn-variant gated \
	--log-file logs/run_history.csv --notes "gated test"
```

- The script mirrors the main pipeline (preprocessing, classical models, optional NN) and appends one row per model to the specified CSV. / 该脚本复用主流程（预处理、经典模型、可选 NN），并将每个模型的指标追加到指定 CSV。
- Each row includes run metadata (`run_id`, timestamp, config JSON, notes) plus metrics (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`). / 每行记录运行元数据（`run_id`、时间戳、配置 JSON、备注）以及指标（`accuracy`、`precision`、`recall`、`f1`、`roc_auc`）。
- Customize `--log-file` to separate experiments or share logs with teammates. / 通过 `--log-file` 选项划分实验或与团队共享记录。

## Project Structure / 项目结构

```
loanpractice/
├── data/                    # Data directory (place your CSV files here) / 数据目录（将 CSV 文件放在这里）
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing / 数据加载和预处理
│   ├── models.py               # ML models / 机器学习模型
│   ├── deep_learning.py        # Neural network models / 神经网络模型
│   └── ensemble.py             # Ensemble methods / 集成方法
├── main.py                  # Main training script / 主训练脚本
├── train.py                 # Quick training script / 快速训练脚本
├── pyproject.toml           # Project metadata & dependencies (uv) / 项目信息与依赖（uv）
├── uv.lock                  # Locked dependency versions / 依赖锁定文件
├── requirements.txt         # Legacy pointer for pip workflows / pip 兼容指引
└── README.md
```

## Models / 模型

### Traditional ML Models / 传统机器学习模型
- **Logistic Regression / 逻辑回归**: Simple baseline model / 简单基线模型
- **Random Forest / 随机森林**: Ensemble of decision trees / 决策树集成
- **Gradient Boosting / 梯度提升**: Sequential boosting method / 序列提升方法
- **XGBoost**: Optimized gradient boosting / 优化的梯度提升
- **LightGBM**: Fast gradient boosting / 快速梯度提升
- **CatBoost**: Handles categorical features well / 良好处理分类特征

### Deep Learning / 深度学习
- **Neural Network / 神经网络**: PyTorch-based multilayer perceptron with dropout and batch normalization / 基于 PyTorch 的带 dropout 和批归一化的多层感知器

### Ensemble Methods / 集成方法
- **Voting / 投票**: Combines predictions via majority vote or probability averaging / 通过多数投票或概率平均组合预测
- **Stacking / 堆叠**: Uses meta-learner to combine base model predictions / 使用元学习器组合基模型预测
- **Weighted Average / 加权平均**: Optimizes weights for combining model probabilities / 优化权重以组合模型概率

## Output / 输出

The prediction script generates a CSV file with: / 预测脚本生成包含以下内容的 CSV 文件：
- `id`: Test sample identifier / 测试样本标识符
- `loan_paid_back`: Predicted repayment outcome (0 or 1) / 预测的还款结果（0 或 1）
- `probability`: Prediction probability (for main.py only) / 预测概率（仅适用于 main.py）

## License / 许可证

MIT License / MIT 许可证
