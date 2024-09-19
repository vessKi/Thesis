import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data with your participants' data
data = {
    'Participant': [
        '99k', '16i', '21n', '79m', '80l', '104p', '12h', '17f', '63e', '19d', '72g', '54c'
    ],
    'Test A Averages': [64, 48.67, 38.33, 36, 43.33, 37.67, 36.33, 35.33, 58, 65.67, 50.67, 32.67],
    'Test B Averages': [57, 72, 40.33, 33.67, 53.67, 78.67, 33, 34.33, 72, 83.33, 90.33, 56.67],
    'Test C Averages': [77.33, 86, 47.67, 38.67, 41.33, 79, 68, 39, 68.67, 57.67, 36.67, 57.33],
}

# Create DataFrame
df = pd.DataFrame(data)

# Reshape the DataFrame for plotting
df_melted = pd.melt(df, id_vars=['Participant'], 
                    value_vars=['Test A Averages', 'Test B Averages', 'Test C Averages'],
                    var_name='Test', value_name='Average')

# Plotting individual scores per participant
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
palette = sns.color_palette("muted", n_colors=3)

# Bar plot for individual scores
bar_plot = sns.barplot(data=df_melted, x='Participant', y='Average', hue='Test', palette=palette, capsize=0.1, errcolor='gray')

# Adding individual data points
for test in data.keys():
    if test != 'Participant':
        sns.stripplot(data=df, x='Participant', y=test, color='black', size=5, jitter=True, dodge=True)

# Add labels and title for individual scores
plt.xlabel('Participant', fontsize=14)
plt.ylabel('CSI Score', fontsize=14)
plt.title('CSI Scores by Participant and Test', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha="right", fontsize=12)

# Calculate total CSI scores for each test
total_csi_a = sum(df['Test A Averages'])
total_csi_b = sum(df['Test B Averages'])
total_csi_c = sum(df['Test C Averages'])

# Customize legend
handles, labels = bar_plot.get_legend_handles_labels()
labels = [f'Test A (Total CSI: {total_csi_a:.2f})', 
          f'Test B (Total CSI: {total_csi_b:.2f})', 
          f'Test C (Total CSI: {total_csi_c:.2f})']
bar_plot.legend(handles, labels, title='Test', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# Calculate the average score for each test
average_scores = {
    'Test': ['Test A', 'Test B', 'Test C'],
    'Average Score': [df['Test A Averages'].mean(), df['Test B Averages'].mean(), df['Test C Averages'].mean()]
}

# Create DataFrame for average scores
df_average = pd.DataFrame(average_scores)

# Plotting overall average scores comparison
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
palette = sns.color_palette("muted", n_colors=3)

# Bar plot for average scores
bar_plot_avg = sns.barplot(data=df_average, x='Test', y='Average Score', palette=palette)

# Add labels and title for average scores
plt.xlabel('Test', fontsize=14)
plt.ylabel('Average CSI Score', fontsize=14)
plt.title('Average CSI Score by Test', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
