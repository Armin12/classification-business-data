import csv
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer
import random


def read_csv_to_dataframe(file_name):
    """Read csv data and save it in a dataframe
    
    This function reads a given csv file and creates a pandas dataframe out of 
    it.
    
    Parameters
    ----------
    file_name : str
         csv file address
    
    Returns
    ----------
    pandas.DataFrame
    
    """
    # Determine required dialects to the read csv file
    with open(file_name, 'r', encoding="utf8") as csvfile:
        sample = csvfile.read(64)
        has_header = csv.Sniffer().has_header(sample)
        print(has_header)
        deduced_dialect = csv.Sniffer().sniff(sample)

    # Read the csv file and store it in a list
    csv_list = []
    with open(file_name, 'r', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, deduced_dialect)

        for row in reader:
            csv_list.append(row)

    # Transform the list into a Pandas dataframe
    csv_df = pd.DataFrame(csv_list[1:], columns=csv_list[0])

    return csv_df


def data_integration(df_list, col_list, rename_dict):
    """Integrate data
    Extract related columns, rename them, and concatenate multiple datasets
    
    Parameters
    ----------
    df_list : list
        List of dataframes
    col_list : list
        List of related columns
    rename_dict : list
        list of rename dictionaries
    
    Returns
    ----------
    pandas.DataFrame
    
    """
    all_related_df = []
    for df, rdict in zip(df_list, rename_dict):
        related_df = df[col_list]
        related_df = related_df.rename(columns=rdict)
        all_related_df.append(related_df)
    concat_df = pd.concat(all_related_df, ignore_index=True, sort=False)
    return concat_df


def sample_dataset(concat_df, no_sample):
    """Sample dataset
    Samples from rows of the input dataset
    
    Parameters
    ----------
    concat_df : pandas.DataFrame
        Input dataframe
    no_sample : int
        Number of samples
    
    Returns
    ----------
    pandas.DataFrame
    
    """
    sample_inds = random.sample(range(len(concat_df)), k=no_sample)
    sampled_dataset = concat_df.iloc[sample_inds]
    return sampled_dataset


def data_cleaning(concat_df, numeric_cols, nonnumeric_cols, threshold=2,
                  imputation_type='SimpleImputer', interpolation_list=None):
    """Cleaning the data
    Convert numeric columns to numeric, delete rows with many missing values, 
    and impute rest of missing values.
    
    Parameters
    ----------
    concat_df : pandas.DataFrame
        Dataframes to clean
    numeric_cols : list
        List of column names which include numerical data for analysis
    nonnumeric_cols : list
        List of column names which include non-numerical data related to 
        properties such as text
    threshold : float
        Threshold for the number of non-nan values to remove a row
    imputation_type : str
        Method of imputation
    interpolation_list : list
        Different columns to interpolate over (over axis=1)
    
    Returns
    ----------
    pandas.DataFrame
    
    """
    # Convert some columns to nunmeric
    concat_df[numeric_cols] = concat_df[numeric_cols].apply(pd.to_numeric)

    # Handling missing values
    concat_df = concat_df.dropna(axis=0, subset=numeric_cols, how='all',
                                 thresh=threshold)  # Deletion: remove rows with less than 2 (threshold) non-NaN elements

    concat_df = concat_df.groupby(
        nonnumeric_cols).mean().reset_index()  # Merge rows with the same non-numeric properties

    if interpolation_list is not None:
        for interpolation_cols in interpolation_list:
            concat_df[interpolation_cols] = concat_df[interpolation_cols].interpolate(method='linear',
                                                                                      limit_direction='both', axis=1)
    else:

        if imputation_type == 'IterativeRandomForest':  # Imputation
            impute_estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
            imp = IterativeImputer(max_iter=20, random_state=0, estimator=impute_estimator)
        elif imputation_type == 'IterativeBayesianRidge':
            impute_estimator = BayesianRidge()
            imp = IterativeImputer(max_iter=20, random_state=0, estimator=impute_estimator)
        elif imputation_type == 'KNNImputer':  # TODO: Does it need normalization?
            imp = KNNImputer(n_neighbors=5, weights="distance")
        else:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')

        concat_df[numeric_cols] = imp.fit_transform(concat_df[numeric_cols])
    # TODO: pipline to test on dataset with missed values

    concat_df = concat_df.dropna()

    return concat_df
