# Machine-Learning-Project-Flight-Delay-Prediction
Use flight features to predict flight delay using logistic regression, decision tree and random forest.

# Objective & Motivation:
The project aims to predict the delay time of airlines based on a series of airline information, specifically, during the COVID-19 pandemic. We are interested in this problem because annoying flight delays and cancellation happen everyday, only more often during the pandemic. These unexpected changes will disrupt people’s travel plans and potentially cause financial losses. If we develop a robust flight delay & cancellation predictive model, it will give much more convenience to airports, flight companies, and passengers to timely adjust the schedule.

# Dataset:
## - Source:
The dataset called “Reporting Carrier On-Time Performance (1987-present)” was extracted from the Bureau of Transportation Statistics in the United States Department of Transportation (1). Similar extraction from this source is also available on Kaggel but that dataset only includes data until 2018 (2; see References 3 and 4 for pre-existing milestones). Since we wish to focus on the airline status during the COVID-19 pandemic, we collected the whole-country data for the latest 12 available months, from Aug. 2020 to Jul. 2021, from the data source. The entire dataset we extracted consists of approximately 6 million rows of flights and 27 columns of features.
## - Target Features:
There are five target features that we are interested in. We will primarily focus on the first three binary features. If time allows, we will also experiment with the two continuous features.

## - Potential Predictor Features:
The rest of 22 features will be evaluated whether they are valid predictors. Some valid ones include day of month, day of week, origin and destination state, scheduled elapsed time of flight, flight distance, etc. Some invalid ones will be excluded either because they are uniform (e.g., year) or because they have direct association with the target features (e.g., actual departure or arrival time).

<img width="812" alt="Screenshot 2021-10-31 at 9 44 14 PM" src="https://user-images.githubusercontent.com/73702692/144119036-50c4792b-9d0d-403c-96bf-d901c5f1f3db.png">
<img width="530" alt="Screenshot 2021-11-30 at 3 01 49 PM" src="https://user-images.githubusercontent.com/73702692/144119312-fad18eea-38d5-478d-97e5-75363236df96.png">
<img width="483" alt="Screenshot 2021-11-30 at 3 00 32 PM" src="https://user-images.githubusercontent.com/73702692/144119334-bf168d2b-42c6-4e06-b382-8239f08e9432.png">
<img width="495" alt="Screenshot 2021-11-30 at 3 00 39 PM" src="https://user-images.githubusercontent.com/73702692/144119353-b02795cf-7df2-46cf-865c-cb9832f9ad6c.png">
<img width="524" alt="Screenshot 2021-11-30 at 3 00 45 PM" src="https://user-images.githubusercontent.com/73702692/144119368-4e2b4087-c040-420b-ba2f-c95540eea2e7.png">
<img width="653" alt="Screenshot 2021-11-30 at 3 00 56 PM" src="https://user-images.githubusercontent.com/73702692/144119380-20b9a6e3-b3b4-4870-88cb-3e23b3146ef6.png">
<img width="850" alt="Screenshot 2021-11-30 at 3 01 10 PM" src="https://user-images.githubusercontent.com/73702692/144119410-82737701-0699-46dd-b779-40ab171b7089.png">
<img width="1237" alt="Screenshot 2021-11-30 at 3 01 18 PM" src="https://user-images.githubusercontent.com/73702692/144119430-56622225-3abc-4e77-9b4c-e1f1235239d8.png">


# Working Plans:
## - Data Preprocessing:
Several steps will be done for preprocessing the data.
1. Some features will be excluded as mentioned in the last section.
2. Categorical features such as airline and state will be converted into one-hot encoding. If some
features turn out to have too many categories (e.g., city), these features may be excluded for
the initial prediction attempts.
3. The numerical features will be evaluated with a pair-wise correlation matrix and all except one
features that highly correlate with each other will be excluded.
4. All numerical features will be centered to zero and scaled to unit variance.
5. Considering that the airline status might undergo changes when increasing numbers of people
were vaccinated, data since May 2021 will be compared with the rest to see if there are noticeable differences. Since our ultimate goal is to predict delay time in the future but not just within the past year, if any significant difference is found, we will only train and test with data since May 2021, which is still sufficiently large with approximately 1.5 million rows.
## - Model Selection:
Multiple models will be tuned and compared to predict the binary target features, including kNN classifier, decision tree classifier, logistic regression, and Naive Bayes classifier. If time allows, we will also tune and compare linear regression and support vector machines for regression to predict the continuous target features.
## - Model Evaluation:
In our case of predicting flight delay, false positives are less important because one’s less likely to be impacted by an on-time flight predicted as delayed, compared to a delayed flight predicted as on time. Therefore, we will primarily evaluate the F-2 score - a weighted harmonic mean of precision and recall with more weight on recall - via 5-fold cross-validation. Other metrics will also be calculated just for references.
