import pandas as pd, numpy as np
from scipy.stats import pearsonr
import glob

rnd = np.random.default_rng(42)
rnd2 = np.random.default_rng(16)
rnd3 = np.random.default_rng(2005)

def brier(probs,label):
        conf_scores = dict()
        for pred,prob in probs.items():
            if int(pred) == label:
                conf_score = (1-prob)**2
                conf_scores[pred] = conf_score
            else:
                conf_score = (0-prob)**2
                conf_scores[pred] = conf_score
            
        non_conformity = np.mean([x for x in conf_scores.values()])

        return conf_scores,non_conformity

l = list()
for doc in glob.glob('output_def/results*.csv'):
    df = pd.read_csv(doc)

    df.probs = df.probs.apply(lambda x:eval(x))
    lab_types = np.sort(df.label.drop_duplicates().to_list())
    print(lab_types)

    gp = df.groupby('annotator_id').label.apply(list).reset_index()

    gp.label = gp.label.apply(lambda x:len(x))


    labs_1 = list()
    labs_2 = list()
    labs_3 = list()
    for _,row in gp.iterrows():
        labs1 = rnd.integers(lab_types[0],lab_types[-1],size=row.label)
        labs_1.extend(labs1)
        labs2 = rnd2.integers(lab_types[0],lab_types[-1],size=row.label)
        labs_2.extend(labs2)
        labs3 = rnd3.integers(lab_types[0],lab_types[-1],size=row.label)
        labs_3.extend(labs3)


    df['label1'] = labs_1
    df['label2'] = labs_2
    df['label3'] = labs_3


    brier_1 = list()
    brier_2 = list()
    brier_3 = list()
    for _,row in df.iterrows():
        a1,b1 = brier(row.probs,row.label1)
        a2,b2 = brier(row.probs,row.label2)
        a3,b3 = brier(row.probs,row.label3)
        brier_1.append(b1)
        brier_2.append(b2)
        brier_3.append(b3)


    df['brier1'] = brier_1
    df['brier2'] = brier_2
    df['brier3'] = brier_3

    

    gp = df.groupby('annotator_id').aggregate({'brier_score':np.average,'brier1':np.average,'brier2':np.average,'brier3':np.average}).reset_index()
    d = dict()
    d['model'] = doc.split('_')[2]
    d['dataset'] = doc.split('_')[-1][:-3]

    r, p = pearsonr(gp['brier_score'], gp['brier1'])
    print(f"brier_score vs brier1: r={r:.4f}, p={p:.4e}")
    d['pearson_brier_brier1'] = r
    d['pvalue_brier_brier1'] = p

    r, p = pearsonr(gp['brier_score'], gp['brier2'])
    print(f"brier_score vs brier2: r={r:.4f}, p={p:.4e}")
    d['pearson_brier_brier2'] = r
    d['pvalue_brier_brier2'] = p

    r, p = pearsonr(gp['brier_score'], gp['brier3'])
    print(f"brier_score vs brier3: r={r:.4f}, p={p:.4e}")
    d['pearson_brier_brier3'] = r
    d['pvalue_brier_brier3'] = p

    r, p = pearsonr(gp['brier1'], gp['brier2'])
    print(f"brier1 vs brier2: r={r:.4f}, p={p:.4e}")
    d['pearson_brier1_brier2'] = r
    d['pvalue_brier1_brier2'] = p

    r, p = pearsonr(gp['brier1'], gp['brier3'])
    print(f"brier1 vs brier3: r={r:.4f}, p={p:.4e}")
    d['pearson_brier1_brier3'] = r
    d['pvalue_brier1_brier3'] = p

    r, p = pearsonr(gp['brier2'], gp['brier3'])
    print(f"brier2 vs brier3: r={r:.4f}, p={p:.4e}")
    d['pearson_brier2_brier3'] = r
    d['pvalue_brier2_brier3'] = p

    l.append(d)



corrs = pd.DataFrame(l)

corrs.to_csv('correlations.csv',index=False)


        



