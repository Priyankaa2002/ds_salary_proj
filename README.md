# Data Science Salary Estimator

A tool that estimates **Data Science salaries** to help professionals negotiate their income when they get a job offer. The model achieves a **Mean Absolute Error (MAE) of ~$11K** on unseen data.

---

## 🚀 Project Overview

This project scrapes job postings from Glassdoor, engineers features from job descriptions, builds regression models, and deploys a **client-facing API** for salary estimation.

- Scraped **1000+ job postings** using Python and Selenium.  
- Engineered features from job descriptions to quantify skills in **Python, Excel, AWS, Spark**.  
- Trained **Linear, Lasso, and Random Forest regressors**, optimized using **GridSearchCV**.  
- Built a **Flask API** for production use.

---

## 🛠 Code & Resources

**Python Version:** 3.13 
**Packages:** `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `selenium`, `flask`, `json`, `pickle`  

##🕸 Web Scraping

Tweaked the scraper to scrape 1000+ Glassdoor job postings. For each job, we collected:

Job title
Salary Estimate
Job Description
Rating
Company Name
Location & Headquarters
Company Size & Founded Date
Type of Ownership
Industry, Sector, Revenue, Competitors

##🧹 Data Cleaning

-After scraping, the following transformations were applied:
-Parsed numeric salary data; removed rows without salary
-Created columns for employer-provided salaries and hourly wages
-Extracted company rating
-Added company_state column and whether job is at HQ
-Calculated company_age from founded date
-Skill columns: Python, R, Excel, AWS, Spark
-Simplified job title & seniority
-Calculated job description length

##📊 Exploratory Data Analysis (EDA)

Explored distributions of numerical variables and value counts of categorical features

Analyzed pivot tables to identify trends across job title, company, and location

##🤖 Model Building

-Converted categorical variables into dummy variables
-Split data: 70% train, 30% test
-Evaluated models using Mean Absolute Error (MAE)

**Models Tested:**

Multiple Linear Regression – Baseline

Lasso Regression – For sparse categorical data

Random Forest Regression – Best performance on sparse data

Model Performance:

Model	MAE ($)
Random Forest	15.58
Linear Regression	20.88
Lasso Regression	19.36

Random Forest clearly outperformed the other models and was chosen for production.

##🚀 Productionization

**Built a Flask API endpoint hosted on a local webserver**

API accepts job listing data and returns estimated salary


