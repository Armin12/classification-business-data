# classification-business-data

A a general framework for getting, cleaning, visualizing, preprocessing, and classification of business data

This framework contains:

- Different strategies to handle missing data
- Standard procedures for cleaning and preprocessing of data
- Different classification estimators (e.g. Random Forest, KNN, Linear/RBF SVM, AdaBoost) for simple and complicated problems
- Different methods for evaluating classification performance
- Different ways to visualize data and showing correlation of features

Functions available in GettingCleaningData.py also handle loading, data integration, and data sampling.

The example in main.py is based on [this](https://opendata.datainfogreffe.fr/explore/?q=Chiffres+Cl%C3%A9s&sort=modified) datasets. T and NE can be any arbitrary column containing numerical values. In each dataset, for each numerical variable, three separate columns are associated which include data from the last three years. For example, you can predict company's sector of activity based on total sales and workforce over several years and compare performance of different estimators.
