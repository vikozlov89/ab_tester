import pandas as pd
import numpy as np



def compare_samples_structures(sample1: pd.DataFrame
                               , sample2: pd.DataFrame
                               , grouping_fields: list
                               , eval_field: str
                               , func=np.mean
                               , sample1_name:str = 'sample_1'
                               , sample2_name:str = 'sample_2')->pd.DataFrame:
    """
    Creates a table with sample structures and aggregation values per group.

    :param sample1: pd.DataFrame
        sample 1 for comparison
    :param sample2: pd.DataFrame
        sample 2 for comparison
    :param grouping_fields: list
        list of categorical field names to group by
    :param eval_field:  str
        field name to apply aggregation
    :param func:
        function for data aggregation
    :param sample1_name: str
        sample 1 name for headers
    :param sample2_name: str
        sample 2 name for headers
    :return: pd.DataFrame
    """
    s1_groups_count = sample1.groupby(grouping_fields)[[eval_field]].count()
    s2_groups_count = sample2.groupby(grouping_fields)[[eval_field]].count()
    s1_groups_shares = s1_groups_count / s1_groups_count.sum()
    s2_groups_shares = s2_groups_count / s2_groups_count.sum()
    s1_groups_ev_field = sample1.groupby(grouping_fields)[[eval_field]].agg(func)
    s2_groups_ev_field = sample2.groupby(grouping_fields)[[eval_field]].agg(func)

    res = pd.concat((s1_groups_count
                     , s2_groups_count
                     , s1_groups_shares
                     , s2_groups_shares
                     , s1_groups_ev_field
                     , s2_groups_ev_field), axis=1)
    res.columns = [f"{sample1_name}_counts"
                  , f"{sample2_name}_counts"
                  , f"{sample1_name}_shares"
                  , f"{sample2_name}_shares"
                  , f"{sample1_name}_func_vals"
                  , f"{sample2_name}_func_vals"]
    return res

def factor_analysis(sample1
                   , sample2
                   , grouping_fields
                   , eval_field
                   , func=np.mean
                   , sample1_name = 'sample_1'
                   , sample2_name = 'sample_2')->pd.DataFrame:
    """

    Performs factor analysis to find out what influenced the target variable.
    FA consists of two steps:

    * calculation of the structure change influence
    * calculation of the variable value change in every group influence

    The sum of all influences should be equal to the difference between aggregated target variable values for
    the whole groups

    The approach to the FA is classic: fixing one of the factors and changing the other one. For example for step one:

    target_variable_weighted_average_group_1 = sample_1_groups_share * sample_1_target_variable_agg_vals_per_groups
    target_variable_weighted_average_group_1_after_str_change = sample_2_groups_share
                                                                * sample_1_target_variable_agg_vals_per_groups
    structure_influence = target_variable_weighted_average_group_1_after_str_change -
                          target_variable_weighted_average_group_1

    :param sample1: pd.DataFrame
        sample 1 for comparison
    :param sample2: pd.DataFrame
        sample 2 for comparison
    :param grouping_fields: list
        list of categorical field names to group by
    :param eval_field:  str
        field name to apply aggregation
    :param func:
        function for data aggregation
    :param sample1_name: str
        sample 1 name for headers
    :param sample2_name: str
        sample 2 name for headers
    :return: pd.DataFrame
    """
    samples_table = compare_samples_structures(sample1
                                               , sample2
                                               , grouping_fields
                                               , eval_field
                                               , func
                                               , sample1_name
                                               , sample2_name).fillna(0)
    val1 = (samples_table.iloc[:, -2] * samples_table.iloc[:, 2]).sum()
    struct_change = (samples_table.iloc[:, -2] * samples_table.iloc[:, 3]).sum() - val1

    cumulative_change = struct_change
    changes_in_levels = []

    struct = samples_table.iloc[:,3].values
    for i in range(samples_table.shape[0]):
        vals = samples_table.iloc[:, -2].copy().values
        vals[:i+1] = samples_table.iloc[:, -1].copy().values[:i+1]
        tmp_val = (vals * struct).sum() - (val1 + cumulative_change)
        cumulative_change += tmp_val
        changes_in_levels.append(tmp_val)



    samples_table['structure_change'] = None
    samples_table.iloc[0, -1] = struct_change
    samples_table['changes_in_levels'] = changes_in_levels
    return samples_table

