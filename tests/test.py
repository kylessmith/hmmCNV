import pandas as pd
bins = pd.read_csv("test_rkx001_100kbins.txt", header=0, index_col=0, sep="\t")

import hmmCNV
tumour_copy = {"rkx001":bins}
loglik, hmm_results = hmmCNV.hmmCNV(tumour_copy, verbose=True)