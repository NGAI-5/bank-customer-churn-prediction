Project Title: **Bank Customer Churn Prediction**
**1. Problem Statement**
A commercial bank is experiencing a 20% annual customer churn rate. This costs the bank significant revenue in acquisition and lost deposits. The bank wants to identify which customers are likely to leave so they can proactively offer retention bonuses or better services before the customer actually closes their account.

**2. Approach / Methodology**
I was given a dataset of 10,000 customers with 14 features (Credit Score, Geography, Gender, Age, Tenure, Balance, etc.).

Step 1: I performed Exploratory Data Analysis (EDA) using Python (Pandas, Matplotlib) to understand the data.

Step 2: I applied Chi-Square tests to check categorical differences (Gender vs Churn, Geography vs Churn) and T-Tests to check numerical differences (Balance vs Churn, Age vs Churn).

Step 3: I built a Logistic Regression model using Scikit-learn, split the data into training (70%) and testing (30%), and evaluated the model's performance.

**3. Solution & Implementation**
A Python script was written to automate the entire process:

Converted categorical variables (Geography, Gender) into numerical format using LabelEncoder.

Trained a Logistic Regression model (sklearn.linear_model.LogisticRegression).

The model achieved an 85% accuracy on the test data, correctly predicting 85 out of 100 customers who would leave.

**4. Results & Business Impact**
The model identified the strongest predictors of churn:

Age: Older customers are 4x more likely to churn.

Balance: Customers with higher balances are significantly more likely to leave.

Geography: Customers in Germany have the highest churn rate compared to France and Spain.

**Recommendation:** The bank should launch a "Loyalty Retention Campaign" targeting customers over 40 with high balances in Germany. By intervening early, the bank could potentially save up to 40% of these at-risk clients, increasing overall customer lifetime value.
