# AI Project

This repository contains implementations of decision tree and random forest models for both classification and regression tasks. Additionally, it includes utility functions for evaluating model performance.

## Structure

- **data**: Contains CSV files (`winequality-red_NO_ALCOHOL.csv` and `winequality-white_NO_ALCOHOL.csv`) with wine quality data.
- **src**: Source code directory containing Python files for model implementations.
  - `BaseModel.py`: Base class for all models.
  - `DecisionTree.py`: Implementation of decision tree models for classification and regression.
  - `RandomForest.py`: Implementation of random forest models for classification and regression.
- **tests**: Test scripts for evaluating model performance.
  - `classification_test.py`: Tests classification models on wine quality data.
  - `regression_test.py`: Tests regression models on wine quality data.
- **utils**: Utility functions directory.
  - `metrics.py`: Contains functions for calculating evaluation metrics like accuracy, precision, recall, etc.
- **AI_project_description_random_forest.pdf**: Description of the AI project focusing on random forests.
- **main.ipynb**: Jupyter notebook demonstrating usage and performance evaluation of implemented models.
- **requirements.txt**: File specifying required Python packages.
- **winequality.md**: Markdown file providing information about the wine quality dataset.

## Model Implementations

- **DecisionTree.py**: Implements decision tree models for both regression and classification tasks. Includes functionalities for fitting the model to the training data and making predictions.
- **RandomForest.py**: Implements random forest models for both regression and classification tasks. Includes functionalities for fitting the model to the training data, making predictions, and aggregating predictions from multiple trees.

## Evaluation

- **classification_test.py**: Script for testing classification models on wine quality data. It evaluates models based on accuracy, precision, recall, and F1-score.
- **regression_test.py**: Script for testing regression models on wine quality data. It evaluates models based on mean absolute error, mean squared error, and root mean squared error.

## Usage

- Clone the repository and navigate to the project directory.
- Ensure Python and required packages are installed (specified in `requirements.txt`).
- Run the test scripts in the `tests` directory to evaluate model performance.
- Refer to the Jupyter notebook (`main.ipynb`) for detailed usage examples and performance evaluation of implemented models.

## Dataset Information

The wine quality dataset contains physicochemical properties of red and white variants of the Portuguese "Vinho Verde" wine. It includes features such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, and sulphates. The target variable is wine quality, which is scored between 0 and 10 based on sensory data. The dataset is available for research purposes and was created by Paulo Cortez et al. [\[Cortez et al., 2009\]](http://dx.doi.org/10.1016/j.dss.2009.05.016).

For more information, refer to `winequality.md`.

