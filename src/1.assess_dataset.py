import pandas as pd


def assess_dataset(folder_path='data/measuring_hatespeech'):
    d = dict()
    majorities = pd.read_csv(f'{folder_path}/majority_types.csv')
    for _,row in majorities.majority_type.value_counts(normalize=True).reset_index().iterrows():
        d[row.majority_type] = row.proportion
    groups = pd.read_csv(f'{folder_path}/top_social_groups.csv')
    tot = sum(groups['count'])
    for item in groups.itertuples():
        d[item.top_social_group] = item.count/tot
    
    false_ratios = pd.read_csv(f'{folder_path}/annotator_false_ratios.csv')
    i = 0
    for _,row in false_ratios.iterrows():
        i +=row.false_ratio * row.n_annotations
    
    d['average_false_ratio'] = i/false_ratios.n_annotations.sum()

    return [d]

x = assess_dataset()

pd.DataFrame(x).to_csv('data/datasets_assessment.csv', index=False)