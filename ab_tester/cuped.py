import pandas as pd


def get_theta(x, metric: str, covariate: str):
    covariance = x[covariate].cov(x[metric])
    variance = x[covariate].var()
    return covariance / variance


def pack_grouping_columns(group_by):
    if isinstance(group_by, str):
        return [group_by]
    try:
        len(group_by)
        return group_by
    except:
        return [group_by]


def calculate_cuped(x, metric: str, covariate: str, theta: float):
    metr_vals = x[metric]
    covariate_vals = x[covariate]
    mean_covariate = covariate_vals.mean()

    return metr_vals - (covariate_vals - mean_covariate) * theta


def make_group_cuped(df, metric: str, covariate: str, test_groups: [list, str], inplace=False):
    tmp = df.copy()
    theta = get_theta(tmp, metric, covariate)
    tmp['cuped'] = tmp.groupby(pack_grouping_columns(test_groups), group_keys=False).apply(calculate_cuped
                                                                                           , metric=metric
                                                                                           , covariate=covariate
                                                                                           , theta=theta)

    if inplace:
        df['cuped'] = tmp['cuped']

    return tmp


if __name__ == '__main__':
    # Пример DF для тестирования ф-ции
    df = pd.DataFrame(
        dict(
            y=(10, 7, 17, 3, 8),
            x=(6, 4, 2, 0, 3),
            test1=('test', 'test', 'control', 'control', 'control'),
            test2=('test', 'test', 'control', 'control', 'control')
        )
    )
    # Вызов ф-ции
    print(make_group_cuped(df, 'y', 'x', ['test1', 'test2']))
