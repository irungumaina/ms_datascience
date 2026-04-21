import numpy as np
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

# --- Method 1: Explicit Calculation (Matches Textbook Math) ---
print("--- Explicit Mathematical Calculation ---")
p_0 = 0.60      # Null hypothesis proportion
n = 200         # Sample size
x = 130         # Number of satisfied customers
alpha = 0.05    # Significance level

# Sample proportion
p_hat = x / n

# Standard Error using the null proportion
SE = np.sqrt((p_0 * (1 - p_0)) / n)

# Z-statistic
z_stat = (p_hat - p_0) / SE

# Two-tailed P-value
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"Sample Proportion (p_hat): {p_hat:.2f}")
print(f"Z-value: {z_stat:.3f}")
print(f"P-value: {p_value:.3f}")

# --- Decision Logic ---
if p_value < alpha:
    print("Conclusion: Reject the Null Hypothesis (Satisfaction differs from 60%)")
else:
    print("Conclusion: Fail to Reject the Null Hypothesis (No significant difference from 60%)")


# --- Method 2: Using Statsmodels Library (Industry Standard) ---
print("\n--- Using Statsmodels Library ---")
# Note: prop_var=0.60 forces the function to use the null proportion for the variance, matching our manual math.
z_stat_sm, p_value_sm = proportions_ztest(count=x, nobs=n, value=p_0, alternative='two-sided', prop_var=p_0)
print(f"Z-value: {z_stat_sm:.3f}")
print(f"P-value: {p_value_sm:.3f}")