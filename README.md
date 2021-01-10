# extended_isolation_forest
Python implementation of the extended isolation forest algorithm for model-free anomaly detection.
Made as a project in the Data Mining course at Radboud University Nijmegen, NL.
Documentation can be found in the file itself.


QUICK START GUIDE:
Import and fit an extended isolation forest via:

	from extended_iForest import iForest
	
	forest = iForest(X)
	
with X being your data as a numpy array. Forest size is set to 100 by default and subsampling size to 256.

Compute an anomaly score via:

	score = forest.anomaly_score(x)
	
with x being one data point. 

Multiple anomaly scores are most easily computed via:

	anom_scores = [forest.anomaly_score(point) for point in x]
