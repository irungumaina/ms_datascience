import numpy as np
import scipy.stats as stats

# 1. Input the data arrays
before_training = np.array([60, 62, 64, 58, 66, 64, 68, 70])
after_training = np.array([65, 67, 69, 60, 71, 66, 72, 74])

# 2. Perform the Paired t-test
# We use ttest_rel which calculates the t-test on TWO RELATED samples
t_statistic, p_value = stats.ttest_rel(after_training, before_training)

# 3. Output the results
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value:     {p_value:.6f}")

# 4. Automate the decision logic
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject the Null Hypothesis.")
    print("The training program significantly improved employee performance.")
else:
    print("\nConclusion: Fail to reject the Null Hypothesis.")
    print("There is no significant evidence that the training improved performance.")