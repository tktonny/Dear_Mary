import numpy as np
import pandas as pd

annotations = pd.read_table(r'flickr30k\results_20130124.token', sep='\t', header=None, names=['image', 'caption'])
print(annotations)

name_index = annotations['image'].str.split('#')
annotations['name'] = name_index.apply(lambda x:x[0])
annotations['index'] = name_index.apply(lambda x:x[1])
annotations['pre'] = annotations['name'].str.split('.').apply(lambda x:x[0]).astype(np.int64)
annotations = annotations.loc[annotations['index']=='2', :]
print(annotations)

annotations.sort_values(by='pre', inplace=True)
annotations = annotations[:1000]
print(annotations)

caption = annotations[['caption', 'name']]
print(caption)

caption.to_csv('caption.txt', sep='\t', header=0, index=0)