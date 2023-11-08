import matplotlib.pyplot as plt

def plot_histogram(data, bins=40):
    # Create the histogram
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram with Statistics')

    # Show the plot
    plt.show()