import numpy as np
from scipy.stats import wilcoxon, norm
import matplotlib.pyplot as plt
# Updated CSI scores from 12 participants for three tests
test_a_csi = np.array([64, 48.67, 38.33, 36, 43.33, 37.67, 36.33, 35.33, 58, 65.67, 50.67, 32.67])
test_b_csi = np.array([57, 72, 40.33, 33.67, 53.67, 78.67, 33, 34.33, 72, 83.33, 90.33, 56.67])
test_c_csi = np.array([77.33, 86, 47.67, 38.67, 41.33, 79, 68, 39, 68.67, 57.67, 36.67, 57.33])

# Function to calculate effect size r
def calculate_effect_size(p_value, n):
    z_value = norm.ppf(1 - p_value / 2)
    return z_value / np.sqrt(n)

# Comparing Test A and Test B
stat, p_value = wilcoxon(test_a_csi, test_b_csi)
effect_size = calculate_effect_size(p_value, len(test_a_csi))
print("Test A vs. Test B - Statistic:", stat, "P-value:", p_value, "Effect Size:", effect_size)

# Comparing Test A and Test C
stat, p_value = wilcoxon(test_a_csi, test_c_csi)
effect_size = calculate_effect_size(p_value, len(test_a_csi))
print("Test A vs. Test C - Statistic:", stat, "P-value:", p_value, "Effect Size:", effect_size)

# Comparing Test B and Test C
stat, p_value = wilcoxon(test_b_csi, test_c_csi)
effect_size = calculate_effect_size(p_value, len(test_b_csi))
print("Test B vs. Test C - Statistic:", stat, "P-value:", p_value, "Effect Size:", effect_size)
data = [test_a_csi, test_b_csi, test_c_csi]

fig, ax = plt.subplots()
ax.boxplot(data, patch_artist=True, labels=['Test A', 'Test B', 'Test C'])
ax.set_title('CSI Scores Across Tests')
ax.set_ylabel('CSI Score')
ax.set_xlabel('Tests')
ax.yaxis.grid(True)
plt.show()