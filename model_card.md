# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classification model built using a RandomForestClassifier from scikit-learn. It is designed to predict whether an individual's income exceeds $50,000 per year based on census data.

## Intended Use
The model is intended to support analysis of income prediction patterns based on demographic and employment-related features.

## Training Data
The model was trained on a version of the U.S. Census Income dataset (`census.csv`). The dataset includes various demographic and categorical features such as education, workclass, marital status, occupation, relationship, race, sex, and native country.

## Evaluation Data
The training set consists of 80% of the dataset, split using a fixed `random_state=42`. The remaining 20% of the dataset was used for evaluation. During evaluation, both the overall model performance and the performance across data slices were computed.

## Metrics
The model was evaluated using the following metrics:
- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

## Ethical Considerations
This model is trained on historical census data, which may contain social and demographic biases. As a result, the model could reflect those biases in its predictions. It should not be used for decisions that affect individuals, such as hiring or lending, without careful review.

## Caveats and Recommendations
The model may not perform well on data that looks different from the training data. It is recommended to test the model on new data before using it.