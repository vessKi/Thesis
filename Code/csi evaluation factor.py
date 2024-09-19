import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('csi_data.csv')

# Display the first few rows to understand the structure
print(data.head())

# Rename the columns appropriately
data.columns = ['CSI_Factors', '99k', '16i', '21n', '79m', '80l', '104p', '12h', '17f', '63e', '19d', '72g', '54c']

# Transpose the ratings DataFrame to get factors as columns
data_transposed = data.set_index('CSI_Factors').T

# Display the transposed data
print(data_transposed.head())

# Calculate means and standard deviations for each factor
means = data_transposed.mean()
std_devs = data_transposed.std()

print("Means:\n", means)
print("Standard Deviations:\n", std_devs)

# Calculate the overall CSI score for each participant
data_transposed['CSI_Score'] = data_transposed.mean(axis=1)

# Calculate the mean and standard deviation of the overall CSI score
mean_csi_score = data_transposed['CSI_Score'].mean()
std_dev_csi_score = data_transposed['CSI_Score'].std()

print("Overall CSI Score Mean:", mean_csi_score)
print("Overall CSI Score Standard Deviation:", std_dev_csi_score)

# Plot boxplot for each CSI factor
plt.figure(figsize=(12, 8))
sns.boxplot(data=data_transposed.iloc[:, :-1])  # Excluding the 'CSI_Score' column
plt.title('Boxplot of CSI Factors')
plt.xlabel('CSI Factors')
plt.ylabel('Scores')
plt.show()

# Plot histogram for overall CSI score
plt.figure(figsize=(10, 6))
sns.histplot(data_transposed['CSI_Score'], kde=True)
plt.title('Distribution of Overall CSI Scores')
plt.xlabel('CSI Score')
plt.ylabel('Frequency')
plt.show()
