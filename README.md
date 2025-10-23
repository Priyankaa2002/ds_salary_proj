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

---

## 🕸 Web Scraping

The project uses a web scraper to collect **1000+ job postings** from Glassdoor. For each job posting, the following information was collected:

- Job Title  
- Salary Estimate  
- Job Description  
- Rating  
- Company Name  
- Location & Headquarters  
- Company Size & Founded Date  
- Type of Ownership  
- Industry, Sector, Revenue  
- Competitors  

---

## 🧹 Data Cleaning

After scraping, the following data cleaning and transformation steps were performed:

- Parsed numeric salary data and removed rows without salary information  
- Created columns for employer-provided salaries and hourly wages  
- Extracted company ratings  
- Added `company_state` column and whether the job is at HQ  
- Calculated `company_age` from founded date  
- Created skill indicator columns: Python, R, Excel, AWS, Spark  
- Simplified job titles and seniority levels  
- Calculated job description length  

---

## 📊 Exploratory Data Analysis (EDA)

- Explored distributions of numerical variables and value counts of categorical features  
- Analyzed pivot tables to identify trends across job title, company, and location  

---

## 🤖 Model Building

### Steps:

1. Converted categorical variables into dummy variables  
2. Split data: 70% training, 30% testing  
3. Evaluated models using **Mean Absolute Error (MAE)**  

### Models Tested:

| Model                     | MAE ($) |
|----------------------------|---------|
| Multiple Linear Regression | 20.88   |
| Lasso Regression           | 19.36   |
| Random Forest Regression   | 15.58   |

> **Random Forest Regression** outperformed the other models and was chosen for production.

---

## 🚀 Productionization

- Built a **Flask API** endpoint hosted on a local web server  
- The API accepts job listing data as input and returns the **predicted salary**  

### Example API Usage:

```python
import requests

url = "http://127.0.0.1:5000/predict"
job_data = {
    "job_title": "Data Scientist",
    "company_name": "ABC Corp",
    "location": "New York, NY",
    "job_description": "...",
    "rating": 4.5,
    "size": "100-500",
    "founded": 2005,
    "ownership": "Private",
    "industry": "IT Services",
    "sector": "Technology",
    "revenue": "$50M-$100M",
    "skills": ["Python", "AWS"]
}

response = requests.post(url, json=job_data)
print(response.json())  # Returns estimated salary


