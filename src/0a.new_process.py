import pandas as pd
from collections import Counter
import numpy as np,glob
from scipy.stats import pearsonr

for doc in glob.glob('output_def/results*.csv'):
    name = doc.split('/')[-1].split('_')[1]
    corpus = doc.split('/')[-1].split('_')[2].split('.')[0]
    path = doc
    df = pd.read_csv(path)

    if 'cade' in path:
        cade = pd.read_csv('data/cade/disaggregated_corpus.csv')

        df = df[['comment_id','text','probs','brier_score']].drop_duplicates().merge(cade[['comment_id','annotator_id','label']])

    df = df.dropna(subset=['label'])

    df.probs = df.probs.apply(lambda x:eval(x))

    annotators = list(set(df.annotator_id.to_list()))


    grouped = df.groupby('comment_id').label.apply(list).reset_index()

    grouped.label = grouped.label.apply(
        lambda x: (
            (lambda c: {k for k, v in c.items() if v == max(c.values())})(Counter(x))
            if len(x) > 2 else None
        )
    )

    grouped = grouped.dropna(subset=['label'])

    grouped = {row.comment_id:row.label for _,row in grouped.iterrows() if row.label is not None}



    l = list()
    for ann in annotators:
        i=0
        j = 0

        d = dict()
        
        tmp = df[df.annotator_id==ann]
        d['brier'] = np.mean([row.probs[str(int(row.label))] for _,row in tmp.iterrows()])
        for _,row in tmp.iterrows():
            i+=1
            labels = grouped.get(row.comment_id)
            if labels and row.label in labels:
                j+=1
            
        d['isolation'] = j/i
        l.append(d)

    forcorr = pd.DataFrame(l)
    corr = pearsonr(x=forcorr.isolation,y=forcorr.brier)

    print(name,corpus)
    print(corr,np.mean(forcorr.brier),np.mean(forcorr.isolation))