import glassdoor_scrapper as gs
import pandas as pd

#path = "C:/Users/lenovo/OneDrive/Desktop/Project_ml/ds_salary_proj/chromedriver"
path = r"C:\Users\lenovo\OneDrive\Desktop\Project_ml\ds_salary_proj\chromedriver.exe"
df = gs.get_jobs('data scientist', 15, False, path, 8)
df.to_csv('glassdoor_jobs.csv', index = False)