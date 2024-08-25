# F1nalyze - Predicting Formula 1 Driver Standings with Machine Learning🚀

## Project Overview

This project, **F1nalyze**, was developed as part of the [F1nalyze - Formula 1 Datathon](https://www.kaggle.com/competitions/f1nalyze-datathon-ieeecsmuj/overview) on Kaggle. The goal of the project was to build machine learning models that accurately predict the finishing positions of Formula 1 drivers based on historical race data.

To explore the code interactively, you can access the project on [Google Colab](https://colab.research.google.com/drive/16dQWIdir3W_m-Wpw0i0w-ioV8ADJCcD9?usp=sharing). This allows you to run the code, tweak the models, and see the results in real time.

Our team, **Frostbiters**, placed 23rd out of 50 teams with a final score of **3.46918** on the leaderboard. Though we didn’t secure a win, the experience was incredibly enriching, offering deep insights into predictive modeling, data preprocessing, and the application of machine learning algorithms.

## Dataset Description

We were provided with three CSV files:
- `train.csv`: Training data used to train the models.
- `test.csv`: Testing data for which we had to predict and submit the output with minimum RMSE.
- `validation.csv`: Validation data used to validate our model's prediction and try to reduce RMSE.

## Data Preprocessing

1. **Handling Missing Data**: 
   - Checked for missing values and occurrences of `\N` in both the train and test datasets.
   - Dropped columns in the training dataset with more than 100 occurrences of `\N`.
   - Filled missing values in the 'status' and 'result_driver_standing' columns with the mode.

2. **Label Encoding**:
   - Used `LabelEncoder` to convert categorical features such as `positionText_x`, `nationality`, `company`, and `status` into numerical values.

3. **Feature Selection**:
   - Selected a specific set of columns based on their relevance for both the train and test datasets.

## Model Training

We experimented with three different machine learning models to predict driver standings:

1. **Decision Tree Classifier**:
   - Initial model used to predict the standings.
   - RMSE on validation data: **3.46918**.

2. **Random Forest Classifier**:
   - Improved model using an ensemble method to enhance prediction accuracy.
   - RMSE on validation data: **3.46918**.

3. **Logistic Regression**:
   - Applied logistic regression for classification after standardizing the data.
   - RMSE on validation data: **3.46918**.

## Model Evaluation

We used Root Mean Squared Error (RMSE) to evaluate the model's performance on the validation dataset. RMSE was calculated for each model:

- **Decision Tree RMSE**: 3.46918
- **Random Forest RMSE**: 3.46918
- **Logistic Regression RMSE**: 3.46918

## Submission

Predictions were made on the test dataset using all three models, and the results were saved in CSV files for submission:

- `dt_predictions.csv`: Decision Tree predictions.
- `rf_predictions.csv`: Random Forest predictions.
- `lr_predictions.csv`: Logistic Regression predictions.

## Leaderboard Performance

Our best entry placed us **23rd** on the leaderboard with a score of **3.46918**. The leaderboard was highly competitive, and the top-performing teams demonstrated impressive predictive accuracy.

## Learnings and Insights

This project provided valuable experience in:

- **Data Preprocessing**: Handling missing data, label encoding, and feature selection.
- **Model Evaluation**: Using RMSE as a key metric for model performance.
- **Machine Learning**: Implementing and comparing different machine learning algorithms for classification.

Despite not winning, the F1nalyze project was a significant learning opportunity, allowing us to enhance our skills in data science and machine learning within the dynamic context of Formula 1 racing.

## Conclusion

The F1nalyze project showcased the potential of machine learning in predicting complex outcomes in sports analytics. We look forward to further refining our models and exploring additional features and techniques to improve prediction accuracy in future endeavors.

## Access the Project on Google Colab

To interact with the code, visit the [Google Colab Notebook](https://colab.research.google.com/drive/16dQWIdir3W_m-Wpw0i0w-ioV8ADJCcD9?usp=sharing).

---

**Team Members**: 
- Nehul Jain
- Ananya Singh

**Team Name**: Frostbiters

## License

This project is licensed under the MIT License.
