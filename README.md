# Loan Prediction Machine Learning Project / 贷款预测机器学习项目

This project implements multiple machine learning and deep learning models to predict loan status based on borrower information.

本项目实现了多种机器学习和深度学习模型，用于根据借款人信息预测贷款状态。

## Features / 功能特性

- **Multiple ML Models / 多种机器学习模型**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost / 逻辑回归、随机森林、梯度提升、XGBoost、LightGBM、CatBoost
- **Deep Learning / 深度学习**: Neural network models using TensorFlow/Keras / 使用 TensorFlow/Keras 的神经网络模型
- **Ensemble Methods / 集成方法**: Voting, Stacking, and Weighted Average ensembles / 投票、堆叠和加权平均集成
- **Automated Preprocessing / 自动预处理**: Handles missing values, categorical encoding, and feature scaling / 处理缺失值、分类编码和特征缩放

## Data Format / 数据格式

The project expects CSV files with the following columns: / 项目需要包含以下列的 CSV 文件：
- `person_age`: Age of the person / 个人年龄
- `person_income`: Annual income / 年收入
- `person_home_ownership`: Home ownership status (RENT, OWN, MORTGAGE, OTHER) / 住房所有权状态（租房、自有、抵押、其他）
- `person_emp_length`: Employment length in years / 就业年限
- `loan_intent`: Purpose of the loan / 贷款目的
- `loan_grade`: Loan grade (A-G) / 贷款等级（A-G）
- `loan_amnt`: Loan amount / 贷款金额
- `loan_int_rate`: Interest rate / 利率
- `loan_percent_income`: Loan amount as percentage of income / 贷款金额占收入的百分比
- `cb_person_default_on_file`: Historical default (Y/N) / 历史违约记录（是/否）
- `cb_person_cred_hist_length`: Credit history length / 信用历史长度
- `loan_status`: Target variable (0 = No Default, 1 = Default) / 目标变量（0 = 无违约，1 = 违约）

## Installation / 安装

```bash
pip install -r requirements.txt
```

## Usage / 使用方法

### Quick Training / 快速训练

```bash
python train.py data/train.csv data/test.csv predictions.csv
```

### Full Training with Options / 带选项的完整训练

```bash
python main.py --train data/train.csv --test data/test.csv --output predictions.csv
```

#### Options / 选项:
- `--train`: Path to training CSV (required) / 训练 CSV 文件路径（必需）
- `--test`: Path to test CSV (required) / 测试 CSV 文件路径（必需）
- `--original`: Path to original loan data CSV (optional) / 原始贷款数据 CSV 文件路径（可选）
- `--output`: Output file for predictions (default: predictions.csv) / 预测输出文件（默认：predictions.csv）
- `--use-nn`: Include neural network models / 包含神经网络模型
- `--ensemble`: Ensemble method - voting, stacking, weighted, or none (default: voting) / 集成方法 - voting、stacking、weighted 或 none（默认：voting）
- `--val-size`: Validation set size ratio (default: 0.2) / 验证集大小比例（默认：0.2）

### Examples / 示例

```bash
# Use voting ensemble (default) / 使用投票集成（默认）
python main.py --train data/train.csv --test data/test.csv --ensemble voting

# Use stacking ensemble / 使用堆叠集成
python main.py --train data/train.csv --test data/test.csv --ensemble stacking

# Include neural network / 包含神经网络
python main.py --train data/train.csv --test data/test.csv --use-nn

# Use best single model / 使用最佳单一模型
python main.py --train data/train.csv --test data/test.csv --ensemble none
```

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
├── requirements.txt         # Dependencies / 依赖项
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
- **Neural Network / 神经网络**: Multi-layer perceptron with dropout and batch normalization / 带 dropout 和批归一化的多层感知器

### Ensemble Methods / 集成方法
- **Voting / 投票**: Combines predictions via majority vote or probability averaging / 通过多数投票或概率平均组合预测
- **Stacking / 堆叠**: Uses meta-learner to combine base model predictions / 使用元学习器组合基模型预测
- **Weighted Average / 加权平均**: Optimizes weights for combining model probabilities / 优化权重以组合模型概率

## Output / 输出

The prediction script generates a CSV file with: / 预测脚本生成包含以下内容的 CSV 文件：
- `id`: Test sample identifier / 测试样本标识符
- `loan_status`: Predicted loan status (0 or 1) / 预测的贷款状态（0 或 1）
- `probability`: Prediction probability (for main.py only) / 预测概率（仅适用于 main.py）

## License / 许可证

MIT License / MIT 许可证
