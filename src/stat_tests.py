import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kruskal, norm
import scikit_posthocs as sp
from itertools import combinations

def assumptions_and_tests(df, val_col, models, variables):
    """
    Checks assumptions for parametric tests and performs Kruskal-Wallis Test for 
    each model and variable if assumptions aren't met.

    This function iterates over multiple models and variables in the dataframe, 
    performing normality checks and variance homogeneity tests. If either 
    assumption is violated, it performs a Kruskal-Wallis test to assess
    differences between groups. The function prints the results of each test, 
    including test statistics, p-values, and interpretations. If significant 
    differences are found through the Kruskal-Wallis test, it displays 
    additional results from a Dunn's test.
    
    The following steps are executed for each model and variable:
    1. Normality and variance homogeneity checks are performed.
    2. If either assumption fails, a Kruskal-Wallis test is performed.
    3. Results from the Kruskal-Wallis test, including the test statistic, 
       p-value, and interpretation, are printed.
    4. If the Kruskal-Wallis test indicates significant differences, Dunn's test 
       results are displayed for further pairwise comparisons.
    
    Parameters:
    df (DataFrame): The input dataframe containing the data.
    val_col (str): The name of the column containing the numeric values to be 
        tested.
    models (list): A list of models to iterate through.
    variables (list): A list of variable names to test across each model.
     
    Returns:
    None: The function prints the results of normality and variance checks,
        Kruskal-Wallis tests, and Dunn's tests (if applicable).

    """
    for v in variables:
        print(f"\033[1mvariable: {v}\033[0m")
        
        for m in models:
            print(f"\033[1mmodel: {m}\033[0m")
            model_df = df[df['model'].str.contains(m)]
            
            normality_check = check_normality_for_groups(model_df, val_col, model_df[v])
            variance_check = check_variance_homogeneity(model_df, val_col, model_df[v])
            if (not normality_check) or (not variance_check):
                kw_test = kruskal_wallis_test(model_df, val_col, v)
                print(f"Kruskal-Wallis Test:")
                print(f"  test statistic: {kw_test['test_statistic']}; p-value: {kw_test['p_value']}")
                i_color = "  \033[38;2;30;200;10m" if (kw_test['Interpretation'] != 'Significant difference between groups') else "  \033[38;5;160m"
                print(i_color + kw_test['Interpretation'] + "\033[0m")
                if kw_test['Interpretation'] == 'Significant difference between groups':
                    display(detailed_dunns_test(model_df, val_col, v))
            print('\n')
        print('---\n')

def check_normality_for_groups(df, val_col, group_col):
    """
    Check the normality of data in each group using the Shapiro-Wilk test.
    
    This function performs the Shapiro-Wilk test for normality on the 
    values in the `val_col` within each group defined by `group_col`. 
    It prints the results for each group and returns a boolean indicating
    whether the normality assumption holds for each group. 
    
    A failed normality test is indicated when the p-value is less than 
    or equal to 0.05.If all groups pass, the function prints "Normality 
    check passed!" and returns `True`. If one or more groups fail, it 
    prints the number of failed groups and returns `False`.
    
    Parameters:
    df (DataFrame): The input dataframe containing the data.
    val_col (str): The name of the column containing the numeric values to be 
        tested for normality.
    group_col (str): The name of the column containing the group labels.
    
    Returns:
    bool: `True` if normality holds for all groups, `False` if at least one 
        group fails the test.
    """
    print("Checking Normality (Shapiro-Wilk Test) for each age group:")
    assumption_failed = 0
    for group in group_col.unique():
        group_values = df[val_col][group_col == group]

        # Perform Shapiro-Wilk test
        stat, p_value = stats.shapiro(group_values)
        if p_value <= 0.05:
            assumption_failed += 1
    if assumption_failed == 0:
        print("  \033[38;2;30;200;10mNormality check passed!\033[0m")
    else:
        print(f"  \033[38;5;160mNormality check failed\033[0m: {assumption_failed} out of {len(group_col.unique())} groups failed")
    return not bool(assumption_failed)


def check_variance_homogeneity(df, val_col, group_col):
    """
    Perform Levene's Test to check for homogeneity of variance across different 
    groups.

    This function assesses whether the assumption of equal variances 
    (homogeneity of variance) is met for different groups in the dataset. The 
    test compares the variances of the groups using their means or medians. It
    prints the test result and whether the variances are equal across groups.
    Additionally, it returns a boolean indicating whether the assumption of 
    equal variance holds. 
    
    A p-value less than or equal to 0.05 indicates a violation of this assumption,
    meaning the variances are significantly different across groups. 

    Parameters:
    df (DataFrame): The input dataframe containing the data.
    val_col (str): The name of the column containing the numeric values to be 
        tested for variance homogeneity.
    group_col (str): The name of the column containing the group labels.

    Returns:
    bool: `True` if the assumption of equal variances holds, `False` otherwise.
    """
    print("Checking Homogeneity of Variance (Levene's Test) for age groups:")
    unique_groups = group_col.unique()
    
    # Create a list of values for each group (grouped by the group column)
    group_values = [df[val_col][group_col == group] for group in unique_groups]

    # Perform Levene's test for homogeneity of variances
    stat, p_value = stats.levene(*group_values)
    
    # Print the results of the test
    print(f"  Levene's test p-value = {p_value}")
    if p_value > 0.05:
        print(f"  \033[38;2;30;200;10mHomogeneity of variances check passed!\033[0m Variances are equal across groups (p > 0.05)")
    else:
        print(f"  \033[38;5;160mHomogeneity of variances check failed\033[0m: variances are NOT equal across groups (p <= 0.05)")
    return p_value > 0.05

def kruskal_wallis_test(df, val_col, group_col):
    """
    Perform a Kruskal-Wallis H test on the given DataFrame and categorical 
    variable.
    
    This function takes in a DataFrame and uses the Kruskal-Wallis H-test to 
    determine whether there are significant differences in the distributions 
    of `query_response` across the groups defined by a categorical variable. It
    returns the test statistic, p-value, and an interpretation of the results.

    Parameters:
    df (DataFrame): The input dataframe containing the data.
    val_col (str): The name of the column containing the numeric values to be 
        tested.
    group_col (str): The name of the column containing the group labels.
    
    Returns:
    dict: A dictionary containing the results of the Kruskal-Wallis test, including:
        - 'test_statistic': The Kruskal-Wallis H statistic.
        - 'p_value': The p-value of the test.
        - 'interpretation': A text string providing the result interpretation, 
          indicating whether there is a significant difference between the groups
          based on the p-value
    """
    # Get categories
    unique_groups = df[group_col].unique()
    
    # Group the data and collect the numeric values for each category
    group_data = [df[df[group_col] == group][val_col] for group in unique_groups]
    
    # Perform Kruskal-Wallis H test
    stat, p = kruskal(*group_data)
    
    result = {
        'test_statistic': stat,
        'p_value': p,
        'Interpretation': 'Significant difference between groups' if p < 0.05 else 'No significant difference between groups'
    }
    return result


def dunns_test(df, val_col, group_col):
    """
    Perform Dunn's test with Bonferroni correction for multiple comparisons.

    This function performs Dunn's test for pairwise comparisons between groups, 
    following a Kruskal-Wallis H-test, and adjusts p-values using the Bonferroni
    correction.

    Parameters:
    df (DataFrame): The input dataframe containing the data.
    val_col (str): The name of the column containing the numeric values to be 
        tested.
    group_col (str): The name of the column containing the group labels.

    Returns:
    DataFrame: A DataFrame containing the p-values of pairwise comparisons with 
        Bonferroni correction.
    """
    # Perform Dunn's test with Bonferroni correction
    dunn_results = sp.posthoc_dunn(df, val_col=val_col, 
                                   group_col=group_col, p_adjust='bonferroni')
    return dunn_results


def detailed_dunns_test(df, val_col=None, group_col=None, p_adjust='bonferroni', sort=True, total_comparisons=1):
    """
    Perform Dunn's test for pairwise comparisons between groups with detailed 
    statistics, including mean and median differences, Z-scores, p-values, and 
    Bonferroni correction for multiple comparisons.
    
    This function calculates Dunn's test for multiple pairwise group comparisons 
    following a Kruskal-Wallis test that indicates significant differences. It 
    computes the Z-value based on rank differences, along with the mean and 
    median differences between groups. The results include p-values with 
    Bonferroni correction to adjust for multiple comparisons.
    
    Dunn's test is a non-parametric post-hoc test used after Kruskal-Wallis to 
    identify which groups differ from each other.
    
    Parameters:
    df (DataFrame or array-like): The input data.
    val_col (str, optional): The name of the column containing the numeric values 
        for the test. Default is `None`.
    group_col (str, optional): The name of the column containing the group labels.
        Default is `None`.
    p_adjust (str, optional): The p-value adjustment method for multiple comparisons. 
        Default is 'bonferroni'.
    sort (bool, optional): If `True`, the data will be sorted by the group and value 
        columns. Default is `True`.
    total_comparisons (int, optional): The total number of comparisons to adjust 
        p-values using the Bonferroni correction. Default is 1.

    Returns:
    pd.DataFrame: A DataFrame containing the results for each pairwise comparison, 
        including:
        - The two groups being compared.
        - The mean and median differences.
        - The Z-score for the difference.
        - The p-value for the comparison.
        - The Bonferroni-adjusted p-value.
        - Flags indicating whether the comparison is significant at p < 0.05 and 
            p < 0.0005 after correction.
    """

    def compare_dunn(i, j):
        """
        Perform the pairwise comparison between two groups using Dunn's test for 
        rank-based differences.
        
        Parameters:
        i (str): The label of the first group.
        j (str): The label of the second group.
        
        Returns:
        tuple: A tuple containing:
            - `mean_diff`: The absolute difference in means between the two groups.
            - `med_diff`: The absolute difference in medians between the two groups.
            - `z_value`: The calculated Z-value for the pairwise comparison.
            - `p_value`: The p-value for the pairwise comparison.
            - `mean_diff_noabs`: The signed mean difference (positive or negative).
            - `med_diff_noabs`: The signed median difference (positive or negative).
        """
        # Compute the standard error and Z-value using ranks, as per Dunn's test
        diff = np.abs(x_ranks_avg[i] - x_ranks_avg[j])
        A = n * (n + 1.) / 12.
        B = (1. / x_lens[i] + 1. / x_lens[j])
        z_value = diff / np.sqrt((A - x_ties) * B)
        p_value = 2 * norm.sf(np.abs(z_value))  # Two-tailed p-value

        # Calculate mean and median differences using actual values
        mean_diff = np.abs(x_means[i] - x_means[j])
        med_diff = np.abs(x_medians[i] - x_medians[j])
        mean_diff_noabs = x_means[i] - x_means[j]
        med_diff_noabs = x_medians[i] - x_medians[j]

        return mean_diff, med_diff, z_value, p_value, mean_diff_noabs, med_diff_noabs

    def __convert_to_df(a, val_col, group_col):
        """
        Convert the input data `a` into a DataFrame, ensuring that the specified 
        columns are used for values and groups.
        
        Parameters:
        a (DataFrame or array-like): The input data, which could either be a 
            DataFrame or an array-like structure.
        val_col (str): The name of the column containing the numeric values for 
            the test.
        group_col (str): The name of the column containing the group labels.
        
        Returns:
        pd.DataFrame: The input data as a DataFrame with the specified value and 
            group columns.
        """
        if isinstance(a, pd.DataFrame):
            return a, val_col, group_col
        else:
            df = pd.DataFrame(a)
            return df, df.columns[-1], df.columns[-2]

    x, _val_col, _group_col = __convert_to_df(df, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_lens = x.groupby(_group_col)[_val_col].count()

    x['ranks'] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col)['ranks'].mean()
    x_means = x.groupby(_group_col)[_val_col].mean()
    x_medians = x.groupby(_group_col)[_val_col].median()

    vals = x.groupby('ranks').count()[_val_col].values
    tie_sum = np.sum(vals[vals != 1] ** 3 - vals[vals != 1])
    x_ties = tie_sum / (12. * (n - 1)) if tie_sum else 0

    results = []
    for i, j in combinations(x_groups_unique, 2):
        mean_diff, median_diff, z_value, p_value, mean_diff_noabs, med_diff_noabs = compare_dunn(i, j)

        # Apply Bonferroni correction for multiple comparisons
        reject_p05 = p_value < (0.05 / total_comparisons)
        reject_p00005 = p_value < ((0.05/1000) / total_comparisons) # Bonferroni correction (0.5/# of p-values)

        results.append({
            group_col + '1': i,
            group_col + '2': j,
            'median_diff': med_diff_noabs,
            'mean_diff': mean_diff_noabs.round(3),
            'Z_score': z_value.round(2),
            'p_value': p_value,
            'p_adj': p_value * total_comparisons,
            'reject_p05': reject_p05,
            'reject_p0005': reject_p00005
        })
    
    results = pd.DataFrame(results)
    results = results.style.applymap(highlight_reject, subset=['reject_p05', 'reject_p0005'])
    return results

def detailed_dunns_test_bivariate(df, val_col, var1, var2, total_comparisons=1):
    result_df = []
    for i in df[var1].unique():
        var1_df = df[df[var1] == i]
        curr_df = detailed_dunns_test(var1_df, val_col, var2, total_comparisons=total_comparisons).data
        curr_df[var1] = i
        curr_df = curr_df.groupby([var1, f"{var2}1", f"{var2}2"]).mean()
        result_df.append(curr_df)
    final_dunns_results = pd.concat(result_df)
    final_dunns_results['reject_p05'] = final_dunns_results['reject_p05'].astype(bool)
    final_dunns_results['reject_p0005'] = final_dunns_results['reject_p0005'].astype(bool)
    final_dunns_results = final_dunns_results.style.applymap(highlight_reject, subset=['reject_p05', 'reject_p0005'])
    
    return final_dunns_results


# Function to color cells
def highlight_reject(val):
    if val is True:
        return 'background-color: lightgreen;'
    elif val is False:
        return 'background-color: lightcoral;'
    return ''