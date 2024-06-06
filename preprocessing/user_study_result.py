import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded file
file_path = './user_study_urban.xlsx'
data = pd.read_excel(file_path)

# Include all models including BADGE
model_names = data.columns

# Calculating means and standard errors for each model including BADGE
means = data[model_names].mean()
std_errors = data[model_names].std(ddof=1) / np.sqrt(len(data))

rgb_colors = [
    [168/255, 168/255, 168/255],
    [120/255, 179/255, 125/255],
    [255/255, 218/255, 130/255],
    [196/255, 100/255, 100/255],
    [135/255, 159/255, 201/255]
]
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 20, 'axes.labelsize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 16})

# Plotting the means with standard errors as a bar graph
plt.figure(figsize=(10, 10))
x = np.arange(len(model_names))
plt.bar(x, means, yerr=std_errors, capsize=5, color=rgb_colors[:len(model_names)], edgecolor='black')

plt.title('User study results')
plt.ylabel('Score')
plt.ylim(1, 7)  # Set y-axis limits to be between 1 and 7
plt.xticks(x, model_names, rotation=0)
plt.show()

print("means")
print(means)
print("std_errors")
print(std_errors)