import large_bvar
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Set frequency of data: quarterly or monthly
monthly = True

if monthly:
    plot_dir = 'plots_monthly/'
    path = "/mnt/data/Data Monthly.xlsx"
else:
    plot_dir = 'plots_SJ/'
    path = "/mnt/data/Data Monthly.xlsx"

os.makedirs(plot_dir, exist_ok=True)
vis = 'on'  # Set to 'off' to hide figures

# Read data
data = pd.read_excel(path, sheet_name="Rep Data")
spec = pd.read_excel(path, sheet_name="Desc Rep")

# Extract the "Dates" column and store it as "dates" table
dates = data['Date'].values

# Remove the "Date" column from "data_table"
data_array = data.drop(columns=['Date']).values

# Transform data according to specification file
data_transformed = transform_data(spec, data_array)
T, n = data_transformed.shape

# Print specification to command window
print(spec.iloc[:, 1:])

# Plot transformed and untransformed data
xl = [dates[0], dates[-1]]

for j_var in range(n):
    f = plt.figure(figsize=(6, 3), dpi=100)

    # Untransformed
    plt.subplot(2, 1, 1)
    plt.plot(dates, data_array[:, j_var], linewidth=1.5)
    plt.xlim(xl)
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
    plt.title(spec.SeriesName[j_var], fontsize=5)

    if monthly:
        plt.gca().tick_params(axis='both', labelsize=5)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d/%y'))
    else:
        plt.gca().tick_params(axis='both', labelsize=5)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y:Q%q'))

    plt.grid()

    # Transformed
    plt.subplot(2, 1, 2)
    plt.plot(dates, data_transformed[:, j_var], linewidth=1.5)
    plt.title('Transformed (' + spec.Transformation[j_var] + ')', fontsize=5)
    plt.xlim(xl)
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())

    if monthly:
        plt.gca().tick_params(axis='both', labelsize=5)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d/%y'))
    else:
        plt.gca().tick_params(axis='both', labelsize=5)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y:Q%q'))

    plt.grid()

    # Save figure
    plt.savefig(plot_dir + 'subplots_' + spec.SeriesID[j_var] + '.png', dpi=600)
    plt.close()
