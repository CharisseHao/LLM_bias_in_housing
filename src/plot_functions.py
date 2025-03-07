import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def single_var_catplot(df, val_col, col_name, title, save_plt = False):
    """
    Creates a boxenplot for a single categorical variable with a custom legend.

    This function generates a seaborn `catplot` (specifically a boxenplot) for a
    categorical variable and overlays the mean values as dots on top of the 
    plot. It also customizes the legend to display both the median line and the 
    mean dots.

    Parameters: 
    df (DataFrame): A DataFrame containing the data to plot. The
        DataFrame must have a column corresponding to the `col_name` and a 
        `val_col` column for the y-values.
    val_col (str): The name of the column containing the numeric values to be 
        tested.
    col_name (str): The name of the categorical column in `df` to be plotted on 
        the x-axis. The function will generate a boxenplot for the values in 
        this column.
    title (str): The title to display at the top of the plot.

    Returns:
    None: This function does not return any value but displays the plot directly.
    """
    var_order = sorted(df[col_name].unique())
    var_stats = df.groupby(col_name)[val_col].agg(['mean', 'median'])

    # Create the boxenplot with specified order
    g = sns.catplot(
        data=df,
        x=col_name,
        y=val_col,
        kind="boxen",
        order=var_order,
        height=4,
        aspect=3,
        orient="v",
        palette=sns.color_palette("Set2"),
        line_kws=dict(linewidth=2, color="#ff1439", alpha=1)
    )
    
    # Add mean dot for each var
    ax = g.ax
    for i, var in enumerate(var_stats.index):
        mean_value = var_stats.loc[var, 'mean']
        ax.scatter(
            x=i,
            y=mean_value,
            color='#39ff14',
            s=25,
            zorder=10,
            label=f"Mean ({var})"
        )

    g.set_xticklabels(rotation=45)
    
    # Add a second legend for the mean and median
    custom_legend = [
        plt.Line2D([0], [0], color="#ff1439", linewidth=2, label="Median"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#39ff14',
                   markersize=8, label='Mean')
    ]
    ax.legend(handles=custom_legend, loc=(1.01,.5), framealpha=0)

    plt.title(title)
    
    if save_plt:
        plt_path = f"plots/{title.replace(' ', '_')}_{col_name}.png"
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        print(f"plot saved successfully as '{plt_path}'")
    
    plt.show()


def multi_var_catplot(df, val_col, col1, col2, title, save_plt = False):
    """
    Creates a boxenplot for two categorical variables with mean and median 
    markers and a custom legend.

    This function generates a seaborn `catplot` (specifically a boxenplot) for 
    the relationship between two categorical variables and a numeric response 
    variable. The mean values are overlaid as green dots, 
    while a custom legend highlights both the median and mean markers.

    Parameters:
    df (DataFrame): A DataFrame containing the data to plot. The 
        DataFrame must include columns for `col1`, `col2`, and `val_col`.
    val_col (str): The name of the column containing the numeric values to be 
        tested.
    col1 (str): The name of the primary categorical column to plot on the x-axis.
    col2 (str): The name of the secondary categorical column to use for 
        grouping (hue).
    title (str): The title to display at the top of the plot.

    Returns:
    None: This function does not return any value but displays the plot directly.
    """
    # Define the order of categories
    col1_order = sorted(df[col1].unique())
    col2_order = sorted(df[col2].unique())

    # Create the boxenplot with specified order
    g = sns.catplot(
        data=df,
        x=col1,
        y=val_col,
        hue=col2,
        kind="boxen",
        order=col1_order,
        hue_order=col2_order,
        height=4,
        aspect=3,
        orient="v",
        palette=sns.color_palette("Set2"),
        legend=True,
        line_kws=dict(linewidth=2, color="#ff1439", alpha=1)
    )

    g.set_xticklabels(rotation=10)

    # Get the Axes object
    ax = g.ax if hasattr(g, 'ax') else g.axes[0,0]

    # Compute the mean and median values
    means = df.groupby([col1, col2])[val_col].mean().reset_index()
    medians = df.groupby([col1, col2])[val_col].median().reset_index()

    # Map categories to positions
    x_pos = np.arange(len(col1_order))
    age_to_x = {age: x for age, x in zip(col1_order, x_pos)}

    width = 0.8  # Default dodge amount
    N = len(col2_order)
    width_per_col2 = width / N
    offsets = np.arange(N) * width_per_col2 - width / 2 + width_per_col2 / 2
    col2_to_offset = {var2: offset for var2, offset in zip(col2_order, offsets)}

    # Calculate x positions for mean and median lines
    means['x'] = means[col1].map(age_to_x) + means[col2].map(col2_to_offset)
    medians['x'] = medians[col1].map(age_to_x) + medians[col2].map(col2_to_offset)

    # Scatter plot for means and medians
    ax.scatter(
        means['x'],
        means[val_col],
        color='#39ff14',
        s=25,
        zorder=10,
        label='Mean'
    )

    # Add a second legend for the mean and median
    # custom_legend = [
    #     plt.Line2D([0], [0], color="#ff1439", linewidth=2, label="Median"),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#39ff14', 
    #                markersize=8, label='Mean')
    # ]
    # ax.legend(handles=custom_legend, loc=(1.01,.05), framealpha=0)

    plt.title(title)

    if save_plt:
        plt_path = f"plots/{title.lower().replace(' ', '_')}_{col1}_{col2}.png"
        plt.savefig(plt_path, dpi=300, bbox_inches='tight')
        print(f"plot saved successfully as '{plt_path}'")
        
    plt.show()
