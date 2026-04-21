# --- PART 1: CHI-SQUARE TEST FOR INDEPENDENCE ---

# 1. Load the built-in dataset
data(UCBAdmissions)

# 2. Create the contingency table
# This step MUST run successfully first. It sums the data by Admission and Gender.
ucb_table <- margin.table(UCBAdmissions, c(1, 2))

# 3. Perform the test and assign it to the variable 'test_independence'
# If this line is not run, you will get the "object not found" error later.
test_independence <- chisq.test(ucb_table)

# 4. Now you can print and view the results
print(test_independence)


# --- PART 2: CHI-SQUARE GOODNESS-OF-FIT TEST ---

# 1. Load the dataset
data(HairEyeColor)

# 2. Create the frequency table for Eye Color
eye_table <- margin.table(HairEyeColor, 2)

# 3. Perform the test and assign it to 'test_gof'
test_gof <- chisq.test(eye_table)

# 4. Print the results
print(test_gof)
