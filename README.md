# Amazon Sales EDA and Modeling
*A data analyst exploratory and modeling project made with Amazon sales and rating data.*

## Overview
This project analyzes Amazon product pricing, sales, and ratings data, applying data preprocessing, exploratory analysis, and modeling techniques to uncover trends and relationships in the dataset.

## Process Summary
In this project, I performed an in-depth analysis of an Amazon dataset to predict discounted prices for products. The process began with data preprocessing and cleaning, where I handled missing values, encoded categorical features, and scaled numerical features. Next, I conducted exploratory data analysis (EDA) to uncover important patterns and relationships in the data. Finally, I built a regression model to predict discounted prices, evaluating the model's performance through metrics like R² and Mean Squared Error. Each phase of the project aimed to enhance the model’s accuracy and interpretability, and the results are showcased in the subsequent sections.

### Data Preprocessing and Cleaning
The dataset contained 1,456 rows of 16 columns. I used `pandas` and `numpy` to help with this process. I determined that the following columns were important for my analysis:

- `category`: The department, sub-department, sub-sub-department, etc. that a product belonged to.
- `discounted_price`: The discounted price in Indian rupees of the product. I chose to interpret this as the final price the product sold for.
- `actual_price`: The original price (before discounts) in Indian rupees of the product..
- `discount_percentage`: The percent of the actual price that was discounted.
- `rating`: The average rating of the product on a 1-5 star scale.
- `rating_count`: The number of ratings that the product received.

From examining the structure of the data, I found some problems with the relevant columns that needed to be addressed before any analysis.

- `category` was presented as a string of the format `department|sub_department|sub_sub_department| . . . |product_type`.
  - I split the string on the `|` separator.
  - The first string became `department`, the second became `sub_department`, and the last string became `product_type`.
    - I felt any additional sub departments would not be as useful for analysis.
- `discounted_price` and `actual_price` were also presented as strings.
  - Both used the Indian rupee symbol at the front of the string.
  - Unit families were separated by commas.
  - Those were removed and both columns were converted to `float64`.
- `rating` had an abnormal value, possibly from a data entry error or something similar.
  - One row contained a `rating` = |, causing the rest of the column to be formatted as a string.
  - Used `pd.to_numeric()` to convert to numerical data, replacing the | with `null` in that column value.
  - Back filled the value with the median of the dataset.
- `rating_count` had two `null` values and commas needed to be removed.
  - Back filled the null value with the median.
 
With no `null` values and the data properly formatted, I added in some add some additional columns that I thought might help with analysis.

- `rating_cat`: The rating system could be broken down into a categorical variable. I used the following scheme:

| Rating Scale | Categorical Rating |
| :-: | :-: |
| 1.0-1.9 | Poor |
| 2.0-2.9 | Fair |
| 3.0-3.9 | Good |
| 4.0-5.0 | Great |

- `est_total_sales`: While not every product gets rated or reviewed, I used `discounted_price` and `rating_count` to estimate the total amount of sales contributed by that product.

### Exploratory Data Analysis (EDA)
For this phase of the project, I used `seaborn` and `matplotlib.pyplot` for visualizations, along with `pandas` for data manipulation. I used `sqlite3` for querying the data, as I find SQL queries to be more efficient for extracting specific information compared to pandas, and this approach is more applicable to real-world data analyst roles, where SQL is commonly used.

I explored the distribution of the quantitative variables using `sns.histplot()` and `sns.boxplot()`. These visualizations, along with the descriptive statistics provided by `pd.describe()` gave me a clearer idea of how the data was distributed.

Using these tools, along with the 1.5 IQR rule, I identified the presence of outliers. However, I chose to retain these outliers in the sample, as they reflect the diverse range of products Amazon sells. The modeling in the next phase would need to account for both higher-priced items and products with less favorable ratings.

I also used the `wordcloud` library on `review_content` to get an idea of what common words appear in product reviews, specifically for `Good` and `Great` rated products.

For more on my insights from this phase of the project, check out the [Key Takeaways](#key-takeaways) section of this document.

### Modeling
In this phase of the project, I focused on predicting `discounted_price` using other variables in the dataset. This prediction could be useful for pricing decisions, helping to determine optimal price points for products. To achieve this, I used the `scikit-learn` library, as it offers robust modeling tools like `LinearRegression` for regression tasks and `OneHotEncoder` for handling categorical variables. These tools helped me build and train models that could estimate the `discounted_price` based on the available features.

For more on my insights from this phase of the project, check out the [Key Takeaways](#key-takeaways) section of this document below.

## Key Takeaways

- The three major departments were `Electronics`, `Computers&Accessories`, and `Home&Kitchen`. Those three dominated the sample, as each department contained over 400 products in the sample, and were the only departments to even reach triple digits in terms of products sold.
- The three major departments all have an average rating of at least 4.0.
- USB cables are the most represented product in the sample, with 233.
  - In general, accessories represent a large portion of the products sold.
- The distributions of `actual_price`, `discounted_price`, and `rating_count` are highly skewed to the right. In terms of the prices, this indicates that the majority of the price points are located on the lower end with a few higher priced items (like laptops, smartphones, and televisions) present in the sample. In terms of ratings, this means that many items receive fewer ratings, with a few receiving a lot more.
- The `discount_percentage` distribution, with a median of 50%, suggests that Amazon products are frequently offered with significant discounts, potentially reflecting sales strategies aimed at attracting more buyers. The slight left skew or symmetry of the distribution implies that while many products are discounted heavily, there is a noticeable proportion of items with smaller or no discounts.
- `rating` is slightly skewed to the left, if not roughly symmetric. The first quartile is 4.0 stars, indicating that a majority of the products are `Great`, suggesting a higher quality of product on Amazon.
- - Using the Pearson Correlation coefficient, there was not much of a correlation between `discounted_price` and `rating`. This suggests that there is more that goes into how a product is rated than just its price. It's likely a combination of the product's price along with its quality and the product type.
- The correlation between `actual_price` and `discounted_price` is very strong, 0.96 specifically. This suggests that `actual_price` aligns well with `discounted_price`.
- To use `actual_price` to predict `discounted_price`, I first used simple linear regression.
  - On the surface, the R-squared of 0.9185 suggested that the model was quite good, but from the high MSE, the scatterplot showing that as `actual_price` increased so too did the variance in `discounted_price`, and the residual plot confirming heteroscedasticity it became clear the linear model was not appropriate.
  - I elected to use a log transformation on both variables. R-squared did decrease by about 1-hundredeth, but from the scatterplot and residual plot, the fit was a lot more accurate.
    - This suggested a power model was more appropriate for predicting `discounted_price` using `actual_price`.
- Adding in the categorical variables `department`, `sub_department`, and `rating_cat` actually improved the performance of the model.
  - Due to the skew in the representation of products in each department and sub department, I used a threshold that each variable had to reach in terms of count. If they didn't meet that threshold, they were placed into an `Other` category. 

## What I Learned

This project marked my first attempt using `OneHotEncoder` for regression analysis. I faced some challenges related to the skew in the representation of the `department` and `sub_department` variables. Specifically, I found that some departments and sub-departments had very little representation in the data. When the model encountered these underrepresented categories during training, it struggled to make accurate predictions due to the lack of sufficient examples.

To address this issue, I decided to classify these less-represented categories as `Other`, effectively creating a catch-all group for rare categories. This approach helped ensure that the model could handle new, unseen categories during testing or future predictions, ultimately improving the model's robustness.

Additionally, I learned the importance of understanding data imbalances and category distributions when preparing data for machine learning. Without addressing these imbalances, the model can easily overfit to the dominant categories, which may lead to poor generalization in real-world applications.

## What I'd Like to Add in the Future

- While I applied a log transformation for better model fit, I would like to explore more complex machine learning algorithms, such as Random Forests, Gradient Boosting, or even neural networks. These models could better capture non-linear relationships and provide more accurate predictions compared to linear regression.
- I did not fully explore the `user_id` column. In the future, I could leverage this data to segment customers into clusters based on their purchasing behaviors, such as products they tend to buy together. This could be useful for developing a more personalized recommendation system.
- Another interesting avenue would be to build a price optimization model using methods like price elasticity of demand. By analyzing how price changes affect sales volume, I could suggest optimal pricing strategies to maximize revenue and improve competitiveness.

## Acknowledgments
The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/).  

**KARKAVELRAJA J.** (n.d.). *Amazon Sales Dataset*. Retrieved December 28, 2024, from [https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset). Licensed under CC BY-NC-SA 4.0.
