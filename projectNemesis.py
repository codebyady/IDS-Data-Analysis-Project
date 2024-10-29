# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
# Import additional libraries for bootstrapping!!!
from scipy.stats import mannwhitneyu

# Load the data
file_path = '/Users/adityataware/Desktop/NYU/IDS/DA Project/movieReplicationSet.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Question 1: Are movies that are more popular rated higher than movies that are less popular?

# Step 1: Calculate the number of ratings (popularity) for each movie
movie_ratings = data.iloc[:, :400]  # Selecting only the first 400 columns which are movie ratings
popularity_counts = movie_ratings.count()  # Count non-missing ratings for each movie

# Step 2: Determine median popularity to split movies into high and low popularity groups
median_popularity = popularity_counts.median()
high_popularity_movies = popularity_counts[popularity_counts >= median_popularity].index
low_popularity_movies = popularity_counts[popularity_counts < median_popularity].index

# Step 3: Calculate average rating for each movie and split into high/low popularity
average_ratings = movie_ratings.mean()  # Average rating for each movie
high_popularity_ratings = average_ratings[high_popularity_movies]
low_popularity_ratings = average_ratings[low_popularity_movies]

# Step 4: Perform a t-test to compare high and low popularity movie ratings
t_stat, p_value = ttest_ind(high_popularity_ratings.dropna(), low_popularity_ratings.dropna())

# Step 5: Display results
print("High Popularity Movies - Average Rating:", high_popularity_ratings.mean())
print("Low Popularity Movies - Average Rating:", low_popularity_ratings.mean())
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Conclusion based on p-value
alpha = 0.005  # Significance level
if p_value < alpha:
    print("There is a significant difference in ratings between high and low popularity movies.")
else:
    print("There is no significant difference in ratings between high and low popularity movies.")


#Enhancing the results!!!
# Define a bootstrap function to calculate confidence intervals manually
def bootstrap_confidence_interval(data1, data2, n_bootstrap=1000, ci=0.95):
    boot_diffs = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(data1.dropna(), size=len(data1.dropna()), replace=True)
        sample2 = np.random.choice(data2.dropna(), size=len(data2.dropna()), replace=True)
        boot_diffs.append(np.mean(sample1) - np.mean(sample2))
    lower_bound = np.percentile(boot_diffs, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(boot_diffs, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

# Step 5 (Alternative): Perform Mann-Whitney U Test for non-parametric comparison
u_stat, u_p_value = mannwhitneyu(high_popularity_ratings.dropna(), low_popularity_ratings.dropna(), alternative='two-sided')

# Step 6: Bootstrap confidence interval for difference in means
bootstrap_ci = bootstrap_confidence_interval(high_popularity_ratings, low_popularity_ratings)

# Step 7: Calculate effect size (Cohen's d)
def cohen_d(x, y):
    return (x.mean() - y.mean()) / np.sqrt(((x.std() ** 2 + y.std() ** 2) / 2))

effect_size = cohen_d(high_popularity_ratings.dropna(), low_popularity_ratings.dropna())

# Output enhanced results
print("Enhanced Results:")
print("Average rating for high popularity movies:", high_popularity_ratings.mean())
print("Average rating for low popularity movies:", low_popularity_ratings.mean())
print("Mann-Whitney U statistic:", u_stat)
print("Mann-Whitney p-value:", u_p_value)
print("Bootstrap 95% Confidence Interval for difference in means:", bootstrap_ci)
print("Effect size (Cohen's d):", effect_size)

if u_p_value < alpha:
    print("There is a significant difference in ratings between high and low popularity movies (non-parametric test).")
else:
    print("There is no significant difference in ratings between high and low popularity movies (non-parametric test).")