A Python framework for testing ensemble classifier algorithms, implemented during my undergraduate Final Year Project.
These ensemble classifiers are specifically designed to tackle class imbalance, so the datasets and evaluation metrics used reflect this. 
This project makes use of scikit-learn, numpy and various other libraries for datasets, base classifiers etc.
All of the algorithms have been created in such a way that they are compatible with the scikit-learn library.

The python implementations that comprise the original work done for this project, are located in ClassifierLibrary.py:
- AdaBoost https://www.sciencedirect.com/science/article/abs/pii/S0031320307001835
- AdaC1, AdaC2, AdaC3, AdaCost https://www.sciencedirect.com/science/article/abs/pii/S0031320307001835
- DataBoost-IM https://dl.acm.org/doi/10.1145/1007730.1007736
- SplitBal, ClusterBal, and MaxDist. https://www.sciencedirect.com/science/article/abs/pii/S0031320314004841

Datasets are loaded in DatasetLoader.py 
Classifiers are configured in JobInitialiser.py

The C45 folder is not my work - sourced from https://github.com/RaczeQ/scikit-learn-C4.5-tree-classifier.
