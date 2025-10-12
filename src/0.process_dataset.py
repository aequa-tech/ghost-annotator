import pandas as pd
from collections import Counter

df = pd.read_csv('data/measuring_hatespeech/corpus.csv')



def compute_relative_frequencies(a_df):
    grouped = a_df.groupby('comment_id').label.apply(list).reset_index()

    grouped['label'] = grouped['label'].apply(
        lambda x: max([(v / sum(Counter(x).values())) for k, v in Counter(x).items()])
    )

    return grouped

def determine_majority_type(score):
    if score==1:
        return 'unanimity'
    elif 0.66 <= score < 1:
        return 'absolute_majority'
    elif 0.5 < score < 0.66:
        return 'majority'
    else:
        return 'no_majority'

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
    grouped['majority'] = grouped.label.apply(lambda x: Counter(x).most_common(1)[0][0])
    a_df = a_df.merge(grouped[['comment_id','majority']], on='comment_id')
    a_df['label_equal_majority'] = a_df.label == a_df.majority

    l = list()

    for item in a_df.annotator_id.unique():
        length = len(a_df[a_df.annotator_id==item])
        ratio = a_df[a_df.annotator_id==item].label_equal_majority.mean()
        l.append({
            'annotator_id': item,
            'false_ratio': ratio,
            'n_annotations': length,
        })
    return pd.DataFrame(l)


relative_frequencies = compute_relative_frequencies(df)
relative_frequencies['majority_type'] = relative_frequencies.label.apply(determine_majority_type)

relative_frequencies.to_csv('data/measuring_hatespeech/majority_types.csv', index=False)


compute_social_groups(df).reset_index().to_csv('data/measuring_hatespeech/top_social_groups.csv', index=False)

compute_false_ratio(df).to_csv('data/measuring_hatespeech/annotator_false_ratios.csv', index=False)