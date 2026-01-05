import pandas as pd,numpy as np
from collections import Counter
import glob

df = pd.read_csv('data/measuring_hatespeech/measuring - violence.csv')


def compute_relative_frequencies(a_df):
    grouped = a_df.groupby('comment_id').label.apply(list).reset_index()
    grouped.label = grouped.label.apply(lambda x: None if len(x)==1 else x)
    grouped = grouped.dropna()
    grouped['relative_label'] = grouped['label'].apply(
        lambda x: max([(v / sum(Counter(x).values())) for k, v in Counter(x).items()])
    )

    return grouped

def determine_majority_type(score):
    if score==1:
        return 'unanimity'
    elif 0.66 <= score < 1:
        return 'qualified_majority'
    elif 0.5 < score < 0.66:
        return 'absolute_majority'
    else:
        return 'relative_majority'

    

def compute_social_groups(a_df):
    grouped = a_df.groupby(['comment_id']).label.apply(list).reset_index()
    grouped['majority'] = grouped.label.apply(lambda x: Counter(x).most_common(1)[0][0])
    a_df = a_df.merge(grouped[['comment_id','majority']], on='comment_id')
    a_df = a_df[a_df.label == a_df.majority]

    sg = a_df.groupby(['comment_id']).social_group.apply(list).reset_index()
    sg['top_social_group'] = sg.social_group.apply(lambda x: Counter(x).most_common(1)[0][0]).reset_index(drop=True)
    sg = sg.top_social_group.value_counts()

    return sg

def compute_false_ratio(a_df):
    grouped = a_df.groupby(['comment_id']).label.apply(list).reset_index()
    grouped.label = grouped.label.apply(lambda x: None if len(x)<=2 else x)
    grouped = grouped.dropna()
    grouped['n_ann'] = grouped.label.apply(lambda x:len(x))

    all_top_labs = list()
    for _,row in grouped.iterrows():
        top_labs = list()
        top_lab = Counter(row.label).most_common()
        max_val = top_lab[0][1]
        for lab in top_lab:
            if lab[1]==max_val:
                top_labs.append(lab[0])
        all_top_labs.append(top_labs)
                
    grouped['majority'] = all_top_labs
    a_df = a_df.merge(grouped[['comment_id','majority']], on='comment_id')

    l = list()
    i = 0
    j = 0
    for item in a_df.annotator_id.unique():
        length = len(a_df[a_df.annotator_id==item])
        for _,row in a_df[a_df.annotator_id==item].iterrows():
            i+=1
            if row.label in row.majority:
                j+=1
        ratio = j/i
        l.append(ratio)
    avg_isolation = 1-np.mean(l)
    std_isolation = np.std(l)
    avg_annotators = np.mean(grouped.n_ann)
    return avg_isolation, std_isolation,avg_annotators




ds_statistics = list()

for doc in ['data/measuring_hatespeech/attitudes - hate speech.csv','data/measuring_hatespeech/cade - acceptability.csv','data/measuring_hatespeech/measuring - violence.csv','data/measuring_hatespeech/davani - offensiveness.csv']:
    keys = [
    "dataset",
    "n_annotators",
    "relative_majority",
    "absolute_majority",
    "qualified_majority",
    "unanimity",
    "avg_penetration",
    "std_penetration",
    "avg_isolation",
    "std_isolation",
]
    d = dict.fromkeys(keys, 0)
    df = pd.read_csv(doc)
    N_LABS = len(df.label.drop_duplicates().to_list())
    def label_distributions(a_list,n_labs=N_LABS):
        unique = set(a_list)
        penetration = len(unique)/n_labs
        return penetration

    name = doc.split('/')[-1].split('-')[-1].strip()[:-4]
    d['dataset'] = name
    relative_frequencies = compute_relative_frequencies(df)
    relative_frequencies['majority_type'] = relative_frequencies.relative_label.apply(determine_majority_type)

    relative_frequencies['penetration'] = relative_frequencies.label.apply(label_distributions)

    for item in relative_frequencies.majority_type.value_counts(normalize=True).reset_index().itertuples():
        d[item.majority_type] =  item.proportion

    d['avg_penetration'] = np.mean(relative_frequencies.penetration)
    d['std_penetration'] = np.std(relative_frequencies.penetration)

    false_ratio = compute_false_ratio(df)
    d['avg_isolation'] = false_ratio[0]
    d['std_isolation'] = false_ratio[1]
    d['n_annotators'] = false_ratio[2]
    ds_statistics.append(d)


pd.DataFrame(ds_statistics).to_csv('data/datasets_assessment.csv', index=False)
#relative_frequencies.to_csv('data/measuring_hatespeech/majority_types.csv', index=False)


#compute_social_groups(df).reset_index().to_csv('data/measuring_hatespeech/top_social_groups.csv', index=False)

