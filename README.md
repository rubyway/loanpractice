# Loan Prediction Machine Learning Project

This project implements multiple machine learning and deep learning models to predict loan status based on borrower information.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Deep Learning**: Neural network models using TensorFlow/Keras
- **Ensemble Methods**: Voting, Stacking, and Weighted Average ensembles
- **Automated Preprocessing**: Handles missing values, categorical encoding, and feature scaling

## Data Format

The project expects CSV files with the following columns:
- `person_age`: Age of the person
- `person_income`: Annual income
- `person_home_ownership`: Home ownership status (RENT, OWN, MORTGAGE, OTHER)
- `person_emp_length`: Employment length in years
- `loan_intent`: Purpose of the loan
- `loan_grade`: Loan grade (A-G)
- `loan_amnt`: Loan amount
- `loan_int_rate`: Interest rate
- `loan_percent_income`: Loan amount as percentage of income
- `cb_person_default_on_file`: Historical default (Y/N)
- `cb_person_cred_hist_length`: Credit history length
- `loan_status`: Target variable (0 = No Default, 1 = Default)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Training

```bash
python train.py data/train.csv data/test.csv predictions.csv
```

### Full Training with Options

```bash
python main.py --train data/train.csv --test data/test.csv --output predictions.csv
```

#### Options:
- `--train`: Path to training CSV (required)
- `--test`: Path to test CSV (required)
- `--original`: Path to original loan data CSV (optional)
- `--output`: Output file for predictions (default: predictions.csv)
- `--use-nn`: Include neural network models
- `--ensemble`: Ensemble method - voting, stacking, weighted, or none (default: voting)
- `--val-size`: Validation set size ratio (default: 0.2)

### Examples

```bash
# Use voting ensemble (default)
python main.py --train data/train.csv --test data/test.csv --ensemble voting

# Use stacking ensemble
python main.py --train data/train.csv --test data/test.csv --ensemble stacking

# Include neural network
python main.py --train data/train.csv --test data/test.csv --use-nn

# Use best single model
python main.py --train data/train.csv --test data/test.csv --ensemble none
```

## Project Structure

```
loanpractice/
├── data/                    # Data directory (place your CSV files here)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── models.py               # ML models
│   ├── deep_learning.py        # Neural network models
│   └── ensemble.py             # Ensemble methods
├── main.py                  # Main training script
├── train.py                 # Quick training script
├── requirements.txt         # Dependencies
└── README.md
```

## Models

### Traditional ML Models
- **Logistic Regression**: Simple baseline model
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential boosting method
- **XGBoost**: Optimized gradient boosting
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Handles categorical features well

### Deep Learning
- **Neural Network**: Multi-layer perceptron with dropout and batch normalization

### Ensemble Methods
- **Voting**: Combines predictions via majority vote or probability averaging
- **Stacking**: Uses meta-learner to combine base model predictions
- **Weighted Average**: Optimizes weights for combining model probabilities

## Output

The prediction script generates a CSV file with:
- `id`: Test sample identifier
- `loan_status`: Predicted loan status (0 or 1)
- `probability`: Prediction probability (for main.py only)

## License

MIT License
