# Analytics Report



Shape immediately after loading: `(891, 15)`



`df.info()`, `df.describe(include='all')`, and `df.shape` are printed during execution.



## Missing values

- `age`: 19.87% missing — median-imputed (5%–30%).

- `embarked`: 0.22% missing — dropped affected rows (<5%).

- `deck`: 77.22% missing — dropped column (>30%; imputation unreliable).

- `embark_town`: 0.22% missing — dropped affected rows (<5%).



## EDA

- `age` IQR outliers: **65**.

- `fare` IQR outliers: **114**.

- Strong correlation: `pclass` and `fare` = **-0.548**. The negative sign means higher-numbered passenger classes generally paid lower fares, so fare also captures part of the socioeconomic class effect.

- Strong correlation: `sibsp` and `parch` = **0.415**. The positive relationship shows that passengers travelling with siblings or spouses were also more likely to travel with parents or children, reflecting family-group travel.

- `survival_by_sex.png`: Women show a substantially higher average survival rate than men. This indicates that sex was strongly associated with access to rescue and should be considered an important predictive feature.

- `survival_by_class.png`: First-class passengers have the highest survival rate, followed by second and third class. The steady decline suggests that cabin location, access to lifeboats, and socioeconomic status influenced survival.

- `fare_by_survival.png`: Survivors generally paid higher fares than non-survivors. Because fares and passenger class are strongly related, this pattern supports the conclusion that class-related access affected survival.

- `age_class_survival.png`: Age, fare, passenger class, and survival overlap rather than forming a perfect decision boundary. Higher-fare passengers contain more survivors, but age and class interactions show that no single feature explains every outcome, supporting multivariate modeling.

- Fare mean=32.097, median=14.454, mode=8.050. Mean above median/mode indicates a right-skewed distribution.



## Survival rates using Boolean masks

|        |   survival_rate |
|:-------|----------------:|
| male   |        0.188908 |
| female |        0.740385 |



|    |   survival_rate |
|---:|----------------:|
|  1 |        0.626168 |
|  2 |        0.472826 |
|  3 |        0.242363 |



| sex    |   pclass |   survival_rate |
|:-------|---------:|----------------:|
| male   |        1 |        0.368852 |
| male   |        2 |        0.157407 |
| male   |        3 |        0.135447 |
| female |        1 |        0.967391 |
| female |        2 |        0.921053 |
| female |        3 |        0.5      |



## EDA standardization check

|      |       age_z |      fare_z |
|:-----|------------:|------------:|
| mean | 2.71749e-16 | 1.39871e-16 |
| std  | 1.00056     | 1.00056     |

Values are approximately mean 0 and standard deviation 1 (minor difference comes from sample-vs-population std conventions).



## Modeling

Class balance: `{0: 0.6162, 1: 0.3838}`.

A stratified split preserves this class ratio in both train and test sets.



### Classifier comparison

|                     |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.7654 |      0.7547 |   0.5797 | 0.6557 | 0.7971 |
| Random Forest       |     0.8156 |      0.8000 |   0.6957 | 0.7442 | 0.8300 |



### Imbalance comparison

|                       |   precision |   recall |     f1 |
|:----------------------|------------:|---------:|-------:|
| baseline              |      0.7931 |   0.6667 | 0.7244 |
| class_weight_balanced |      0.7297 |   0.7826 | 0.7552 |
| SMOTE_train_only      |      0.7397 |   0.7826 | 0.7606 |

`SMOTE_train_only` produced the best F1 balance on this split; SMOTE was applied only after training preprocessing.



### Random Forest tuning

Best parameters: `{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 100}`

OOB score: **0.8272**



### Regression metrics

| Model             |     MAE |    RMSE |     R2 |   Adjusted R2 |
|:------------------|--------:|--------:|-------:|--------------:|
| Linear Regression | 20.8977 | 30.5328 | 0.3975 |        0.3617 |

The residual plot suggests heteroscedasticity; correlation between absolute residual size and prediction is 0.546.



## Final model comparison

Classification metrics and regression metrics are reported as separate groups because they measure different tasks and are not directly comparable.

### Classification models

|                     |   accuracy |   precision |   recall |     f1 |    auc |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.7654 |      0.7547 |   0.5797 | 0.6557 | 0.7971 |
| Random Forest       |     0.8156 |      0.8000 |   0.6957 | 0.7442 | 0.8300 |

### Regression model

| Model             |     MAE |    RMSE |     R2 |   Adjusted R2 |
|:------------------|--------:|--------:|-------:|--------------:|
| Linear Regression | 20.8977 | 30.5328 | 0.3975 |        0.3617 |



## Recommendation

Deploy **Random Forest** because it has the highest test F1 (0.7442) while achieving AUC 0.8300. F1 balances precision and recall, making it more informative than accuracy alone for the minority survived class. The final choice should still be monitored for subgroup fairness and drift.

Reloaded raw-input prediction check succeeded: `0`.