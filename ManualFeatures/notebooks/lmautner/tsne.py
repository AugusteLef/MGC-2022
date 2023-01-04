from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5)

z_df = pd.read_csv('z50.csv', index_col=0)
z = np.array(z_df)

z_vis = tsne.fit_transform(z)

pd.DataFrame(z_vis).to_csv('z50_vis.csv')
