# classification-business-data

A a general framework for getting, cleansing, visualizing, preprocessing, and classification of business data.  

This framework contains:

- Getting, integrating, and sampling data
- Multiple strategies to handle missing data
- Standard data cleaning and preprocessing procedures
- Different methods to visualize data and determine correlations in features
- Multiple estimators for classification (e.g. Random Forest, KNN, Linear/RBF SVM, AdaBoost)
- Different metrics for classification performance evaluation


The example in main.py is based on datasets available in [this](https://opendata.datainfogreffe.fr/explore/?q=Chiffres+Cl%C3%A9s&sort=modified) link. T and NE can be any arbitrary column containing numerical values. In each dataset, three separate columns are associated with each numerical variable that include data from the last three years. An example analysis can be performed to predict company's sector of activity based on total sales and workforce over several years and compare performance of different estimators.
