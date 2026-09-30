# House Price Prediction Using Machine Learning

A machine learning project that predicts house prices from area characteristics such as average income, house age, number of rooms, number of bedrooms, and population. It covers the full workflow: exploratory data analysis, preprocessing, training and comparing three regression models, evaluation, and a command-line application for making predictions.

**Best result:** Linear Regression, **R² = 0.9180**, **MAE ≈ $80,879** on a held-out test set of 1,000 samples.

## Table of contents

- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [CLI usage](#cli-usage)
- [Methodology](#methodology)
- [Results](#results)
- [Challenges and solutions](#challenges-and-solutions)
- [Future improvements](#future-improvements)

## Problem statement

Estimating the price of a house is a classic regression task. Given a handful of characteristics of the area where a house is located, the model predicts its sale price:

| Feature | Description |
|---|---|
| `Avg. Area Income` | Average income of residents in the area |
| `Avg. Area House Age` | Average age of houses in the area |
| `Avg. Area Number of Rooms` | Average number of rooms |
| `Avg. Area Number of Bedrooms` | Average number of bedrooms |
| `Area Population` | Population of the area |
| **`Price`** | **Target: house sale price** |

## Dataset

[USA Housing Dataset](https://www.kaggle.com/datasets/vedavyasv/usa-housing/data) (Kaggle).

- 5,000 records, 7 columns
- No missing values
- The `Address` column is excluded: it is free text and not useful for this regression setup

## Tech stack

- **Language:** Python
- **Libraries:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook
- **Tools:** Visual Studio Code, GitHub

## Project structure

```
.
├── data/
│   └── USA_Housing.csv      # dataset
├── src/
│   ├── main.py              # data loading, preprocessing, training, evaluation, CLI
│   └── diagram.ipynb        # EDA, model comparison and plots
├── tests/
├── report/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Getting started

**Requirements:** Python 3.10+

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application from the repository root
python src/main.py
```

The dataset is loaded with a relative path (`data/USA_Housing.csv`), so run the program from the repository root. The project also includes `pyproject.toml` and `uv.lock`, so it can be set up with [uv](https://docs.astral.sh/uv/) as well.

To explore the analysis, open `src/diagram.ipynb` in Jupyter or VS Code.

## CLI usage

Running `main.py` opens an interactive menu:

```
=== House Price Prediction ===
1. Show Dataset Information
2. Train Model
3. Predict House Price
4. Exit

Select option:
```

A typical session: look at the dataset information, train the model, then enter the area's average income, house age, number of rooms, number of bedrooms, and population to get a predicted price.

## Methodology

### 1. Exploratory data analysis

The notebook includes the price distribution, feature distributions, a correlation heatmap, and scatter plots of income, population, and rooms against price.

Correlation of each feature with `Price`:

| Feature | Correlation | Interpretation |
|---|---|---|
| Avg. Area Income | 0.640 | Strongest relationship |
| Avg. Area House Age | 0.453 | Moderate positive |
| Area Population | 0.409 | Moderate positive |
| Avg. Area Number of Rooms | 0.336 | Weak to moderate positive |
| Avg. Area Number of Bedrooms | 0.171 | Very weak; bedrooms alone explain little of the price |

### 2. Preprocessing

- Drop the `Address` column
- Split into features `X` and target `y = Price`
- Train/test split: 80% / 20% (4,000 / 1,000 samples), `random_state=42`
- Features are scaled before training the models in the notebook

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3. Models

| Model | Idea | Settings |
|---|---|---|
| Linear Regression | Fits a linear combination of features by minimizing squared error | defaults |
| Decision Tree Regressor | Splits data by feature thresholds; predicts the average of a leaf | `max_depth=10`, `random_state=42` |
| Random Forest Regressor | Ensemble of trees trained on random subsets; averages their predictions | `n_estimators=100`, `random_state=42` |

### 4. Evaluation metrics

All models are evaluated on the same held-out test set:

- **MAE** — average prediction error in dollars
- **RMSE** — like MAE but penalizes large errors more
- **R²** — share of the price variance explained by the model

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **80,879** | **100,444** | **0.9180** |
| Random Forest Regressor | 94,645 | 120,252 | 0.8825 |
| Decision Tree Regressor | 133,997 | 170,977 | 0.7624 |

**Linear Regression** gave the lowest errors and the highest R², so it was selected as the final model. Random Forest performed well but was less accurate, and the Decision Tree was the weakest.

Additional diagnostics in the notebook for the final model:

- predicted vs. actual prices (points follow the ideal line closely)
- residuals vs. predicted values and the distribution of residuals
- feature coefficients: `Avg. Area Income` has the largest effect, while `Avg. Area Number of Bedrooms` has almost none

## Challenges and solutions

| Challenge | Solution |
|---|---|
| Learning the basics of machine learning, regression models, and evaluation metrics | Free video playlists on YouTube and selected chapters of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd edition) by Aurélien Géron |
| Problems with Python libraries | Created a virtual environment (`.venv`) |
| File paths differing between Linux and Windows | Used a relative path (`data/USA_Housing.csv`) instead of a full path |

## Future improvements

- Feature engineering from the `Address` column (for example, extracting the state)
- Testing Gradient Boosting and XGBoost
- Cross-validation and hyperparameter tuning instead of a single train/test split
