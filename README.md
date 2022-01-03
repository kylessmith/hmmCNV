# HMM copy number variation calling

[![Build Status](https://travis-ci.org/kylessmith/hmmCNV.svg?branch=master)](https://travis-ci.org/kylessmith/hmmCNV) [![PyPI version](https://badge.fury.io/py/hmmCNV.svg)](https://badge.fury.io/py/hmmCNV)
[![Coffee](https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee&color=ff69b4)](https://www.buymeacoffee.com/kylessmith)

This is a Python package for Copy Number Variation
detection using an HMM as implemented in [ichorCNA][code].

All citations should reference to [original paper][paper].

## Install

If you dont already have numpy and scipy installed, it is best to download
`Anaconda`, a python distribution that has them included.  
```
    https://continuum.io/downloads
```

Dependencies can be installed by:

```
    pip install -r requirements.txt
```

PyPI install, presuming you have all its requirements installed:
```
	pip install hmmCNV
```

## Usage

```python
from hmmCNV import hmmCNV

# Call CNVs for bin data
loglik, hmm_results = hmmCNV.hmmCNV(tumour_copy, verbose=True)
```

[code]: https://github.com/broadinstitute/ichorCNA
[paper]: https://www.nature.com/articles/s41467-017-00965-y