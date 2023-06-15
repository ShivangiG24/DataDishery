# DataDishery 👩🏽‍🍳
## Abstract ✍️
This is an in-depth analysis of publicly-available data sourced from [food.com](https://www.food.com), conducted by Shivangi Gupta and Shreya Sudan for DSC 80 (The Practice and Application of Data Science) at UC San Diego. It encompasses answering a prediction problem using linear regression on various quantitative data. In order to see our previous work with the dataset which involved a series of rigorous analytical procedures, including data cleansing, exploratory data analysis, data visualization, and hypothesis testing, check out [this link](https://shivangig24.github.io/DataDish/). 

# Introduction 😇

Hey! We chose the `recipes` dataset because we are both passionate about cooking and felt slightly nostalgiac about working with the Great British Bake Off in DSC 10 last spring. It was a wonderful choice, to be honest, since we had a fun time reading through very creative recipe names, instructions, and reviews. We also worked with this dataset in Project 3.

The Recipes dataframe has 83782 rows × 12 columns and the Interactions dataframe has 731927 rows × 5 columns. Note that the bulk of the project utilizes a merged version of both of these dataframes, resulting in a dataframe that has 83782 rows × 13 columns.

Throughout this project, we focused mainly on the `n_steps`, `n_ingredients`, and `minutes` column, as we wanted to learn more about how the number of ingredients in a recipe and the time it takes to execute the recipe affects the number of steps. We also encoded categorical columns such as `description` and `name` into quantitative data that can be used in a regression model.

The prediction problem which guided our data exploration was "predict the number of steps in recipes."


Here are the components of our project:

1. Framing the Problem
    - Pre-Processing Recipes and Interactions
    - Exploratory Data Analysis
    - Prediction Problem and Evaluation Metric
2. Baseline Model
    - Feature Description
    - Model Description & Performance
3. Final Model
    - New Features
    - Choosing Hyperparameters
    - Final Model vs Baseline Model
4. Fairness Analysis
    - Permutation Test Set Up and Results

A glimpse at the original dataframes:

`interactions`

|    |    user_id |   recipe_id | date       |   rating | review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---:|-----------:|------------:|:-----------|---------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 |    1293707 |       40893 | 2011-12-21 |        5 | So simple, so delicious! Great for chilly fall evening. Should have doubled it ;)<br/><br/>Second time around, forgot the remaining cumin. We usually love cumin, but didn't notice the missing 1/2 teaspoon!                                                                                                                                                                                                                                                                                                                                                                                                                 |
|  1 |     126440 |       85009 | 2010-02-27 |        5 | I made the Mexican topping and took it to bunko.  Everyone loved it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|  2 |      57222 |       85009 | 2011-10-01 |        5 | Made the cheddar bacon topping, adding a sprinkling of black pepper. Yum!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|  3 |     124416 |      120345 | 2011-08-06 |        0 | Just an observation, so I will not rate.  I followed this procedure with strawberries instead of raspberries.  Perhaps this is the reason it did not work well.  Sorry to report that the strawberries I did in August were moldy in October.  They were stored in my downstairs fridge, which is very cold and infrequently opened.  Delicious and fresh-tasting prior to that, though.  So, keep a sharp eye on them.  Personally I would not keep them longer than a month.  This recipe also appears as #120345 posted in July 2009, which is when I tried it.  I also own the Edna Lewis cookbook in which this appears. |
|  4 | 2000192946 |      120345 | 2015-05-10 |        2 | This recipe was OVERLY too sweet.  I would start out with 1/3 or 1/4 cup of sugar and jsut add on from there.  Just 2 cups was way too much and I had to go back to the grocery store to buy more raspberries because it made so much mix.  Overall, I would but the long narrow box or raspberries.  Its a perfect fit for the recipe plus a little extra.  I was not impressed with this recipe.  It was exceptionally over-sweet.  If you make this simple recipe, MAKE SURE TO ADD LESS SUGAR!                                                                                                                          |

`recipes`

| name                                 |     id |   minutes |   contributor_id | submitted   | tags                                                                                                                                                                                                                        | nutrition                                    |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                    |   n_ingredients |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil'...]                                                  | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour'] |               9 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl , sift together the flours and baking powder', 'set aside', 'in another mixing bowl , blend together the sugars , margarine , and salt until light and fluffy'...] | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                    |              11 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce'...]                                                                                                                                                                                                                                                                                                                              | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']          |               9 |

# Framing the Problem 🖼️
## Pre-Processing `recipes` and `interactions`

1. Loaded in both csv files and read them. 
2. Left-merged on recipes and replaced all 0 ratings with np.NaN because these were recipes that were possibly not reviewed yet, which is intrinsically different than a recipe being given a 0-star rating. 
3. Then, we found the average rating per recipe as a Series by calculating the mean rating after grouping by the `recipe_id` and using the transform function.
4. We added the average rating as a column to `recipes`.
5. Checked the datatypes of columns within the Recipes dataframe to make sure they matched what they should be logically (ie `minutes` should be an int). 
6. When we were examining the datatypes closely, we noticed that some columns appeared to look like lists but were actually strings. For these such columns (`ingredients`, `tags`, `nutrition`), we removed the opening and closing parenthesis as well as split by string to ensure the inherent data remained the same and now fit the proper format. Note that for the nutrition column specifically, we separated each index within a singular list and created new columns in the dataframe to match each index with the value it represented. For example, calories was the first element, total fat was the second, and so on.

The first few rows of cleaned `recipes`:

| name                                 |     id |   minutes |   contributor_id | submitted   | tags                                                                                                                                                                                                                                                                  | nutrition                                    |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                                               |   n_ingredients |   average_rating |   calories |   total fat |   sugar |   sodium |   protein |   saturated fat |   carbohydrates |   cooking time (minutes) | is 60-minutes-or-less   |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|-----------------:|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|-------------------------:|:------------------------|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ["'60-minutes-or-less'", " 'time-to-make'", " 'course'", " 'main-ingredient'", " 'preparation'", " 'for-large-groups'", " 'desserts'", " 'lunch'", " 'snacks'", " 'cookies-and-brownies'", " 'chocolate'", " 'bar-cookies'", " 'brownies'", " 'number-of-servings']"] | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |        10 | ['heat the oven to 350f and arrange the rack in the middle', 'line an 8-by-8-inch glass baking dish with aluminum foil'...]                                                  | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ["'bittersweet chocolate'", " 'unsalted butter'", " 'eggs'", " 'granulated sugar'", " 'unsweetened cocoa powder'", " 'vanilla extract'", " 'brewed espresso'", " 'kosher salt'", " 'all-purpose flour']"] |               9 |                4 |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 |                       60 | True                    |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ["'60-minutes-or-less'", " 'time-to-make'", " 'cuisine'", " 'preparation'", " 'north-american'", " 'for-large-groups'", " 'canadian'", " 'british-columbian'", " 'number-of-servings']"]                                                                              | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |        12 | ['pre-heat oven the 350 degrees f', 'in a mixing bowl , sift together the flours and baking powder', 'set aside', 'in another mixing bowl , blend together the sugars , margarine , and salt until light and fluffy'...] | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ["'white sugar'", " 'brown sugar'", " 'salt'", " 'margarine'", " 'eggs'", " 'vanilla'", " 'water'", " 'all-purpose flour'", " 'whole wheat flour'", " 'baking soda'", " 'chocolate chips']"]              |              11 |                5 |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 |                       60 | True                    |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ["'60-minutes-or-less'", " 'time-to-make'", " 'course'", " 'main-ingredient'", " 'preparation'", " 'side-dishes'", " 'vegetables'", " 'easy'", " 'beginner-cook'", " 'broccoli']"]                                                                                    | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |         6 | ['preheat oven to 350 degrees', 'spray a 2 quart baking dish with cooking spray , set aside', 'in a large bowl mix together broccoli , soup , one cup of cheese , garlic powder , pepper , salt , milk , 1 cup of french onions , and soy sauce'...]                                                                                                                                                                                                                                                                                                                              | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ["'frozen broccoli cuts'", " 'cream of chicken soup'", " 'sharp cheddar cheese'", " 'garlic powder'", " 'ground black pepper'", " 'salt'", " 'milk'", " 'soy sauce'", " 'french-fried onions']"]          |               9 |                5 |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 |                       60 | True                    |


## Exploratory Data Analysis

1. As part of our EDA, we checked for NaN values in the columns and found that 'name' had 1 missing value, 'description' had 70 missing values, and our new 'average_rating' column now had 1002 missing values (which used to be 0's).
2. We then proceeded to conduct further EDA and create bivariate plots to better understand the correlation between our 3 columns of interest (`n_ingredients`, `n_steps`, `minutes`). Scroll down to see our plots and explanatory comments.
3. We used the .describe() method on the aforementioned columns to learn their summary statistics and other relevant information like outliers. This information was useful to us later on when we were deciding how to transform certain columns.

<iframe src="assets/mins_bivariate.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="assets/ingredients_bivariate.html" width=800 height=600 frameBorder=0></iframe>

## Prediction Problem and Evaluation Metric

**Prediction Problem:** Predict the number of steps in recipes.

**Response Variable**: `n_steps`

**Regressors**: `n_ingredients`, `minutes`

**Evaluation Metric**: R^2

After seeing the scatterplots above, we thought it would be interesting to predict the number of steps using the number of ingredients and minutes features using linear regression. We thought `n_steps` would be an interesting response variable to analyze since prior to building our regression model, we thought there would be a somewhat clear relationship between the number of ingredients and minutes it takes to create a recipe. Both of our regressors would also be known at the time of prediction:
- `n_ingredients`: when making a recipe, you need to know which and how many ingredients are needed
- `minutes`: when making a recipe, you tend to have a general idea of how long it'll take based on precedence or information learnt from others
Also, it's worth mentioning that all of these values are given to us by the user (the creator of the recipe) at the same time. Lastly, our evaluation metric used throughout our model was R^2 since linear regression conventionally deals with either R^2 or RMSE as evaluation metrics since the data is quantitative and such values can be mathematically computed.

# Baseline Model 🧩
## Feature Description
The model created is a linear regression model, where the objective is to predict the number of steps (`n_steps`) in a recipe based on a number of different features. These features are:

`minutes`: The time it takes to complete the recipe. This is a quantitative variable.

`n_ingredients`: The number of ingredients in the recipe. This is a quantitative variable.

`name`: The name of the recipe. This is a nominal variable that is transformed into a quantitative variable through the NameLengthTransformer, which computes the length of the recipe name.

`description`: The description of the recipe. This is a nominal variable that is transformed into a quantitative variable through the RecipeLengthTransformer, which computes the length of the description.

Hence, after transformation, all the features used in the model are quantitative.

The ColumnTransformer object in the pipeline is used to apply the NameLengthTransformer and RecipeLengthTransformer to the appropriate columns. These transformers turn the nominal variables (`name` and `description`) into quantitative ones by taking their length. The `remainder='passthrough'` argument implies that the remaining columns (`minutes` and `n_ingredients`) are left as they are.

## Model Description & Performance
The data is split into a training set and a test set, with 75% of the data used for training and 25% for testing. The model is trained on the training set and then its performance is evaluated on the test set using the coefficient of determination R^2, which measures the proportion of the variance in the dependent variable that is predictable from the independent variables.


The model is considered "good" if the R^2 score is close to 1. This would mean that our model can explain a large proportion of the variance in the number of steps in the recipes. If the R^2 score is far from 1, then the model is considered to have poor predictive performance. Our trained model has an R^2 of ~0.2117 which is therefore pretty poor. This is likely due to the fact that the quantitative features are not transformed and are simply being "passed through," so we aren't quite capturing how `minutes` and `n_ingredients` impact or influence the number of steps.

To improve this model, we will experiment with other feature transformations and encodings for the future model.

# Final Model 🏆 
## New Features

After looking at the columns above, we considered the following transformation for our new features:
- standardized, quantile-transformed, degree 5 `minutes`: Since `minutes` had a lot of variety (high variance), we could split it in bins and analyze each bin separately; larger values lead to different predictions than smaller ones. Lastly, we chose to use StandardScaler() to standardize all values. Then, we applied PolynomialFeatures to make our prediction model fit better on our trained data (refer to our 'Choosing Hyperparameter' subheading). Lastly, we used StandardScaler() to standardize all values.
- standardized, degree 5 `n_ingredients`: We applied PolynomialFeatures and StandardScaler() for the same reason as above.
- standardized, degree 5 logarithmic `calories`: We chose to incorporate `calories` as it's a quantitive feature that we omitted in the baseline model, and after analyzing the relationship between it and `n_steps` using a scatterplot, we saw that `n_steps` shared a fairly exponential relationship with `calories`. We chose to perform a log transformation on it. Lastly, we chose to use StandardScaler() to standardize all values.

## Choosing the Hyperparameters

1. `minutes`:
    - **Degree 5 (Polynomial) Transformation:** After running an iterative search for the best polynomial degree for `minutes`, we concluded degree=5 would be the best hyperparameter (see errs_df output below). This is because it had the lowest average validation RMSE, which determines best hyperparameters during cross-validation. 
    - **Quantile Transformation:** Looking at the scatterplot of minutes vs steps, we saw there was a very exponential relationship so we thought doing a quantile transformation would fix it since we wanted to reduce the impact of outliers and make the model better at capturing variability of data!
    
2. `calories`: 
    - **Logarithmic Transformation:** A logarithmic transformation made the model's prediction better fit the data linearly, however we still observed an initially positive, then flat, then negative relationship between the variables. This led us to subsequently perform a polynomial transformation (see below).

<iframe src="assets/file-name.html" width=800 height=600 frameBorder=0></iframe>

    - **Degree 5 (Polynomial) Transformation:** After running an iterative search for the best polynomial degree for `calories`, we concluded degree=5 would be the best hyperparameter (see errs_df output below). This is because it had the lowest average validation RMSE, which determines best hyperparameters during cross-validation.
    
3. `n_ingredients`:
    - **Degree 5 (Polynomial) Transformation:** After running an iterative search for the best polynomial degree for `n_ingredients`, we concluded degree=5 would be the best hyperparameter (see errs_df output below). This is because it had the lowest average validation RMSE, which determines best hyperparameters during cross-validation.

`errs_df`: hyperparameters-search for best PolynomialFeatures degree

| Validation Fold   |   Deg 5 |   Deg 6 |   Deg 7 |   Deg 8 |   Deg 9 |   Deg 10 |   Deg 11 |
|:------------------|--------:|--------:|--------:|--------:|--------:|---------:|---------:|
| Fold 1            | 5.37032 | 5.37056 | 5.36986 | 5.36968 | 5.36662 |  5.35708 |  5.35033 |
| Fold 2            | 5.37503 | 5.37711 | 5.39059 | 5.55004 | 5.88816 |  5.61469 |  5.6313  |
| Fold 3            | 5.45436 | 5.45467 | 5.45146 | 5.44858 | 5.44651 |  5.44244 |  5.44442 |
| Fold 4            | 5.377   | 5.37668 | 5.37291 | 5.36846 | 5.36577 |  5.35764 |  5.34654 |
| Fold 5            | 5.37231 | 5.37563 | 5.39157 | 5.42175 | 5.44563 |  5.41682 |  5.36766 |

## Final Model vs Baseline Model

The final model was constructed using **Linear Regression**, a supervised learning algorithm utilized for predicting a continuous outcome variable (or dependent variable) based on one or more predictor variables (or independent variables). The outcome variable we aim to predict in this scenario is 'n_steps', representing the number of steps required to complete a recipe.

The final model incorporates an additional feature, `calories`, that wasn't present in the baseline model. The baseline model included the features `minutes`, `n_ingredients`, `name`, and `description`. We assumed that recipes with higher caloric content might generally be more complex and thus require more preparation steps which is why we chose to include `calories` in the final model.

Furthermore, the final model applies more sophisticated data transformations and feature engineering techniques in comparison to the baseline model:

- Minutes: The `minutes` feature undergoes a quantile transformation, polynomial expansion (degree 5), and then standardization. The quantile transformation makes the distribution of `minutes` more uniform, enhancing the performance of linear regression. Polynomial expansion allows the model to capture non-linear relationships between `minutes` and `n_steps`. Lastly, standardization ensures that `minutes` contributes to the model on the same scale as other features.

- Number of Ingredients: The feature `n_ingredients` is transformed with a polynomial expansion (degree 5) and then standardized for similar reasons as mentioned for `minutes`.

- Calories: `calories` is also  transformed with a polynomial expansion (degree 5) and then standardized for similar reasons as mentioned for `minutes`. In addition to the aforementioned transformations, `calories` also undergoes a log transformation to handle the wide range of values it can take, helping to enhance linear regression's performance with this feature; this means that we are able to better discern a linear relationship between the 2 variables.

These transformations were chosen and refined based on exploratory data analysis and model validation techniques, which represent a **manual, iterative hyperparameter-selection process**. Specifically, the degree 5 polynomial was chosen after testing various degrees and selecting the one that led to the best performance on the validation set. 

Upon comparing the final model with the baseline model, the final model exhibits superior performance, as indicated by a higher R^2 score. A higher R^2 score means that the model can explain more of the variation in `n_steps`, leading to more accurate predictions. This improvement suggests that the inclusion of the `calories` feature and the application of more complex data transformations in the final model substantially enhance prediction accuracy.

# Fairness Analysis ⚖️
## Permutation Set Up and Results

Our experiment investigates the difference in prediction accuracy between two groups of recipes:

**Group X:** recipes with a high caloric content

**Group Y:** recipes with a low caloric content

We define "high caloric content" as a caloric value higher than the median (305.40), and conversely, "low caloric content" as a caloric value lower or equal to the median. We chose the median as the threshold, because `calories` in `recipes` has many outliers, so the median is a better representative for the centre of its distribution. 

**Evaluation Metric:** R^2, a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.

We define our null hypothesis and alternative hypothesis as follows:

**Null Hypothesis:** There is no difference in the prediction accuracy (as measured by R^2) of our model between high calorie and low calorie recipes.

**Alternative Hypothesis:** There is a difference in the prediction accuracy (as measured by R^2) of our model between high calorie and low calorie recipes.

**Test Statistic:** The absolute difference in R^2 scores between the two groups

**Significance level:** 0.05, a commonly chosen value in statistical testing

We performed a permutation test to generate the p-value. In this test, we randomly reassigned the recipes to the two groups, computed the R^2 scores, and calculated the absolute difference for each permutation. We repeated this process 1000 times.

<iframe src="assets/fairness.html" width=800 height=600 frameBorder=0></iframe>

**p-value:** 0.332, representing the proportion of permutations in which the absolute difference in R^2 scores exceeded the observed absolute difference in R^2 scores

**Conclusion:** Given the p-value of 0.332, which is greater than the significance level of 0.05, we fail to reject the null hypothesis. In other words, we do not have sufficient evidence to conclude that there is a significant difference in the prediction accuracy of our model between high calorie and low calorie recipes. This suggests that the caloric content of a recipe does not have a significant impact on the prediction accuracy of our model for the number of steps in a recipe.
