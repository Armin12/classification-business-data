print(__doc__)

# Author: Armin Najarpour Foroushani -- <armin.najarpour-foroushani@polymtl.ca>

###################### Import Libraries ##########################

from GettingCleaningData import (read_csv_to_dataframe, data_integration, 
                                 sample_dataset, data_cleaning)

from ExploratoryDataAnalysis import (two_labels_data_scatter_plotter, 
                                     label_averaged_data_scatter_plotter,
                                     df_time_series_plotter)

from ClassificationFuncs import train_test_preparator, data_classifier

from matplotlib import rc


###################### Set fonts ##########################
font = {'family' : 'sans-serif','weight' : 'bold','size'   : '25'}
rc('font', **font)  # pass in the font dict as kwargs

###################### Main ##########################

### Load the data

KeyFigures2019 = read_csv_to_dataframe('KeyFigures2019.csv')
KeyFigures2018 = read_csv_to_dataframe('KeyFigures2018.csv')
KeyFigures2017 = read_csv_to_dataframe('KeyFigures2017.csv')

### Integrating data

KeyFigures = [KeyFigures2019, KeyFigures2018, KeyFigures2017]

col_list = ['Name', 'BA', 'City', 'T1', 'T2', 'T3', 'NE1', 'NE2', 'NE3']

rename_dict2019 = {'T1' : 'T2019', 'T2' : 'T2018', 'T3' : 'T2017',
               'NE1' : 'NE2019', 'NE2' : 'NE2018', 'NE3' : 'NE2017'}

rename_dict2018 = {'T1' : 'T2018', 'T2' : 'T2017', 'T3' : 'T2016', 
               'NE1' : 'NE2018', 'NE2' : 'NE2017', 'NE3' : 'NE2016'}

rename_dict2017 = {'T1' : 'T2017', 'T2' : 'T2016', 'T3' : 'T2015', 
               'NE1' : 'NE2017', 'NE2' : 'NE2016', 'NE3' : 'NE2015'}

rename_dict = [rename_dict2019, rename_dict2018, rename_dict2017]

integrated_data = data_integration(KeyFigures, col_list, rename_dict)

### Create a sampled dataset

integrated_data = sample_dataset(integrated_data, no_sample=100000)

### Cleaning the data

numeric_cols = ['T2019', 'T2018', 'T2017', 'T2016', 'T2015', 
                'NE2019', 'NE2018', 'NE2017', 'NE2016', 'NE2015']

nonnumeric_cols = ['Name', 'BA', 'City']

interpolation_list = [['T2019', 'T2018', 'T2017', 'T2016', 'T2015'], 
                      ['NE2019', 'NE2018', 'NE2017', 'NE2016', 'NE2015']]

cleaned_data = data_cleaning(integrated_data, numeric_cols, nonnumeric_cols, 
                             interpolation_list=interpolation_list)

### Exploratory data analysis

two_labels_data_scatter_plotter(cleaned_data, colx='NE2017', coly='T2017', 
                                col_label='BA', label1='7022Z', label2='9329Z', 
                                xmin=-10, xmax=400, ymin=0, ymax=1e8)

label_averaged_data_scatter_plotter(cleaned_data, colx='NE2017',
                                    coly='T2017', col_label='BA',
                                    xmin=-10, xmax=400, ymin=-1e6, ymax=1e8)

dftime = cleaned_data[['T2019', 'T2018', 'T2017', 'T2016', 'T2015']].iloc[1:20, :]
df_time_series_plotter(dftime)


### Classification

X_train, X_test, y_train, y_test = train_test_preparator(cleaned_data, numeric_cols, 
                                                         col_label='BA', min_freq = 400)

classification_performance = data_classifier(X_train, X_test, y_train, y_test, 
                                             estimator_type='Random Forest', 
                                             performance_measure='f1_macro')

