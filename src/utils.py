# ================================== IMPORTS ==================================
# Standard Library Imports
import logging
import os
import pickle
import random
import re
import time
import warnings
from itertools import combinations_with_replacement
from typing import Optional, Union

# Third-Party Imports
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from flaml import AutoML
from flaml.automl.data import get_output_from_log
from molfeat.trans import MoleculeTransformer
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from scikitplot.metrics import plot_confusion_matrix, plot_precision_recall, plot_roc
from scipy import linalg
from scipy.stats import fisher_exact
import statsmodels.stats.multitest as smt
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Configuration
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

###############################################################################
# ====================== Cell Painting Data Preprocessing =====================
###############################################################################

def load_cellpainting_data(profiles_dir, drop_nan_cols=True):

    logging.info('Loading the well-level Cell Painting profiles from all plates...')
    # List all plate directories from the /profiles_dir
    plate_dirs = list(os.listdir(profiles_dir))
    # Get a list of all CSV file names containig the plate Cell Painting profiles
    filenames = [profiles_dir+'/'+plate+'/profiles/mean_well_profiles.csv' for plate in plate_dirs]
    # Import all CSV files and concatenate them into a Pandas dataframe
    profiles_df = pd.concat((pd.read_csv(file) for file in filenames), ignore_index=True)

    if drop_nan_cols == True:
        # Create an empty list to save the name of the columns containing only NaN values
        nan_cols = []
        for col in profiles_df:
            if profiles_df[col].isnull().values.all():
                nan_cols.append(col)
                logging.info(f"{col}: {profiles_df[col].isnull().sum()} NaN values - column removed")
        # Remove the columns containing only NaN vaues
        profiles_df = profiles_df.drop(nan_cols, axis=1)

    # Return the Cell Painting dataframe
    return profiles_df

def plot_plate_layout_effects(raw_data):
        # Create a new figure
        plt.figure(figsize=(14, 10))

        # Select 4 random plates
        selected_plates = random.sample(list(raw_data['Metadata_Plate'].unique()), 4)

        # Iterate through the selected plates to check whether they have plate effects 
        for index, plate in enumerate(selected_plates, start=1):
            # Create a dataframe containig the plate data in the origianl spatial format
            plate_df = raw_data.loc[raw_data['Metadata_Plate'] == plate][['Metadata_Well', 'Cells_Number_Object_Number']]
            plate_df['Metadata_Well_row'] = plate_df['Metadata_Well'].str.extract(r'([a-zA-Z]+)')
            plate_df['Metadata_Well_column'] = plate_df['Metadata_Well'].str.extract(r'(\d+)')
            plate_df = plate_df.pivot(index='Metadata_Well_row', columns='Metadata_Well_column', values='Cells_Number_Object_Number')
             
            # Create subplots
            plt.subplot(2, 2, index)
             
            # Plot the heatmap
            sns.heatmap(plate_df, annot=False, vmin=-50, vmax=80)
            plt.xlabel('Well column')
            plt.ylabel('Well row')
            plt.title('Heatmap of cell count on Plate %d' % plate)

        plt.tight_layout()
        plt.show()

class WhiteningTransform(object):
    def __init__(self, controls):
        # Set a correction parameter 
        correct_param = 10**np.log(1/controls.shape[0]) 
        # Compute the mean of the controls to perform centering of the data (i.e., mean 0)
        self.mean = controls.mean(axis=0)
        # Compute the whitening matrix W on DMSO control wells using PCA whitening
        self.whithening_matrix(controls - self.mean, correct_param)
        
    def whithening_matrix(self, X, lambda_):
        '''
        Function that computes the sphering matrix W using the PCA 
        whitening as the chosen natural whitening procedure.
        '''
        # Compute the covariance matrix C of X
        C = (1/X.shape[0]) * np.dot(X.T, X)
        # Perform the eigendecomposition of C to obtain the eigenvectors V and eigenvalues s 
        s, V = linalg.eigh(C)
        # Calculate the square root of the eigenvalues s
        D = np.diag( 1. / np.sqrt(s + lambda_) ) # apply the correction parameter to avoid dividing by 0
        # Compute W as the dot product of the squared eigenvalues s and the transposed eigenvectors V
        W = np.dot(V, D) # first apply a change of basis
        W = np.dot(W, V.T)
        self.W = W

    def transformation(self, X):
        ''' 
        Function that performs the whitening transformation applying the 
        previously computed sphering transform.
        '''
        return np.dot(X - self.mean, self.W)
    
def mean_aggregation(well_data, feature_columns):
    # Compute the average value of every feature per dose-treatment combination 
    cpd_dose_mean = well_data.groupby(['Metadata_broad_sample', 'Metadata_mmoles_per_liter'])[feature_columns].mean(numeric_only=True).reset_index()
                                        # keep both metadata columns in the resulting dataframe
    # Return the resulting dataframe
    return cpd_dose_mean

def plot_data_transformation(raw_data, processed_data):
    # Create subplots
    fig, axs = plt.subplots(6, 2, figsize=(12, 20))

    # Select 6 features to print 
    features_to_print = ['Cells_AreaShape_Center_X', 'Cells_Correlation_Correlation_ER_AGP', 'Cells_Correlation_Correlation_Mito_RNA', 
                         'Cells_Correlation_Costes_AGP_DNA', 'Cells_Correlation_K_DNA_AGP', 'Nuclei_Number_Object_Number']
    
    if not set(features_to_print).issubset(set(list(raw_data.columns))):
        # Select 6 random features
        features_to_print = random.sample(list(raw_data.columns), 6)

    # Iterate over the selected features
    for i in range(len(features_to_print)):
        feature = features_to_print[i]
        # Plot the original distribution
        axs[i, 0].hist(raw_data[feature], bins=400, color='skyblue')
        axs[i, 0].set_title('Original %s distribution' %feature)
        # Plot the transformed distribution
        axs[i, 1].hist(processed_data[feature], bins=400, color='r')
        axs[i, 1].set_title('Transformed %s distribution' %feature)
        axs[i, 1].set_xlim(-20,20)

    # Set labels
    for ax in axs.flat:
        ax.set(xlabel='Value', ylabel='Frequency')

    plt.tight_layout()
    plt.show()

def plot_compound_concentrations(trt_data):
    # Summary statistics of morphological feature values
    logging.info('··· Compound concentrations = %.4f ± %.4f ···' %(trt_data['Metadata_mmoles_per_liter'].mean(), 
                                                                   trt_data['Metadata_mmoles_per_liter'].std()))
    logging.info(trt_data['Metadata_mmoles_per_liter'].describe())

    # Plot the distribution of compound concentrations in an histogram
    plt.figure(figsize=(10,6))
    plt.hist(trt_data['Metadata_mmoles_per_liter'], bins=40, color='skyblue', edgecolor='black') # set the number of bins to 20
    plt.xlabel('Compound concentration (uM)')
    plt.ylabel('Frequency')
    plt.title('Distribution of compound concentrations')
    plt.show()

def data_preprocessing(raw_data, well_level_preprocessing, treatment_aggregation,
                       check_plate_layout_effects=True, check_feature_distribution=True, check_compound_concentrations=True):
    
    # Define the function arguments and their expected data types and allowed values
    args_type = {'well_level_preprocessing': str, 'treatment_aggregation': str,
                 'check_plate_layout_effects': bool, 'check_feature_distribution': bool, 'check_compound_concentrations': bool}
    
    args_values = {'well_level_preprocessing': ['TVN'], 'treatment_aggregation': ['mean']}
    
    # Validate argument data types and allowed values
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value.")
        # Check allowed values for specific arguments
        if argument in args_values:
            value = locals()[argument]
            if value not in args_values[argument]:
                raise ValueError(f"'{argument}' must be one of {args_values[argument]} values.")

    # Get two lists of metadata and feature columns
    meta_cols = [col for col in raw_data if col.startswith('Metadata')]
    feature_cols = [col for col in raw_data if not col.startswith('Metadata')]

    if check_plate_layout_effects == True:
        logging.info('Checking the plate-layout effects on 4 random plates...')
        plot_plate_layout_effects(raw_data)
        
    if well_level_preprocessing == 'TVN':
        logging.info('Preprocessing the well-level profiles by applying the TVN transform...')
        # Select the rows containig data from perturbation and negative control wells
        controls = raw_data.loc[raw_data['Metadata_broad_sample_type'] == 'control']
        trt = raw_data.loc[raw_data['Metadata_broad_sample_type'] == 'trt']

        # Create an instance of the WhiteningTrasnform class using the control data
        TNV_transform = WhiteningTransform(controls[feature_cols])
        logging.info('--- Computation of the TVN transform from control data ---')

        # Apply the computed sphering transform on perturbation data
        trt_transformed = TNV_transform.transformation(trt[feature_cols])
        logging.info('--- Linear transformation of the treatment data using the TVN transform ---')

        well_data = pd.concat([trt[meta_cols].reset_index(drop=True),
                               pd.DataFrame(data=trt_transformed, columns=feature_cols).reset_index(drop=True)], axis=1)

    # Get the total number of compounds
    logging.info('··· Total number of compounds = %d ···' %len(well_data['Metadata_broad_sample'].unique()))

    # Group by compound identifier and count unique compound dose values
    cpd_dose_counts = well_data.groupby('Metadata_broad_sample')['Metadata_mmoles_per_liter'].nunique()
    logging.info('··· Compounds tested at more than one concentration: ···')
    logging.info(cpd_dose_counts[cpd_dose_counts > 1].index.tolist())

    if treatment_aggregation == 'mean':
        logging.info('Preprocessing the treatment-level profiles by performing mean aggregation...')
        # Perform mean aggregation
        treatment_data = mean_aggregation(well_data, feature_cols)

    if check_feature_distribution == True:
        logging.info('Checking the distribution of 6 features before and after their transformation...')
        # Plot the density distribution of 6 features
        plot_data_transformation(raw_data, treatment_data)
    
    if check_compound_concentrations == True:
        logging.info('Analysing the compound concentrations in the dataset...')
        # Inspect the compound concentrations
        plot_compound_concentrations(treatment_data)

    # Return the resulting dataframe
    return treatment_data

def get_correlated_features(processed_data, feature_columns):
    # Create an empty set to store the correlated features
    correlated_features = set()

    # Compute the Pearson correlation coefficient for every pair of features
    corr_matrix = processed_data[feature_columns].corr(method='pearson', numeric_only=True)

    # Get the indices of the upper triangular part of the dataframe and select the corresponding values
    indices = np.triu_indices(n=corr_matrix.shape[0], m=corr_matrix.shape[1], k=1)
    corr_values = corr_matrix.values[indices]

    # Plot an histogram to show the distribution of correlation coefficients
    plt.figure(figsize=(10,6))
    plt.hist(corr_values, bins=40, color='skyblue', edgecolor='black') # set the number of bins to 40
    plt.axvline(x=0.8, color='red', linestyle='--', linewidth=1) # positive threshold
    plt.axvline(x=-0.8, color='red', linestyle='--', linewidth=1) # negative threshold
    plt.xlabel('Pearson Correlation Coefficient (PCC)')
    plt.ylabel('Frequency')
    plt.title('PCC distribution of pairs of morphological features')
    plt.show()

    # Create an empty set to store the pairs of correlated features
    correlated_pairs = set()

    # Set the (absolute) threshold to 0.8
    threshold = 0.80
    
    # Find pairs of highly correlated features
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]): # iterate only through the upper triangular half of the correlation matrix
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pair = (corr_matrix.index[i], corr_matrix.columns[j])
                correlated_pairs.add(pair)
    
    # Find the feature with largest mean absolute correlation with all features for each pair 
    for pair in correlated_pairs:
        mean_corr1 = corr_matrix[pair[0]].abs().mean(numeric_only=True)
        mean_corr2 = corr_matrix[pair[1]].abs().mean(numeric_only=True)
        if mean_corr1 > mean_corr2:
            correlated_features.add(pair[0])
        else:
            correlated_features.add(pair[1])

    # Return the correlated features
    return correlated_features

def get_invariant_features(processed_data, feature_columns):
    # Create an empty set to store the invariant features
    invariant_features = set()

    # Compute the variance of the morphological features
    feature_var = processed_data[feature_columns].var(numeric_only=True) 

    # Plot feature variances in a histogram
    plt.figure(figsize=(10,6))
    plt.hist(feature_var, bins=40, color='skyblue', edgecolor='black') # set the number of bins to 40
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.title('Variance distribution of morphological features')
    plt.show()

    # Set the threshold to the 1st percentile of the total variance 
    sorted_var = sorted(feature_var) 
    index = int(0.01 * len(sorted_var)) # index of the 1st percentile
    threshold = sorted_var[index] 

    # Identify the features with a variance lower than the threshold
    for i in range(len(feature_var)):
        if feature_var[i] < threshold:
            invariant_features.add(feature_var.index[i])

    # Return the invariant features
    return invariant_features

def feature_selection(processed_data, correlated_features=False, invariant_features=False):

    # Define the function arguments and their expected data types
    args_type = {'correlated_features': bool, 'invariant_features': bool}

    # Validate argument data types
    logging.info('Validating argument data types...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value")
    
    # Get a list of feature columns
    feature_cols = [col for col in processed_data if not col.startswith('Metadata')]
    
    # Create an empty set to store the features to be removed
    features_to_remove = set() # to avoid duplicates 

    # Add to the set the already identified invariant and almost invariant features
    features_to_remove.update(processed_data.columns[processed_data.columns.str.endswith('Count') | processed_data.columns.str.endswith('EulerNumber') | processed_data.columns.str.contains('Correlation_Manders')])
    logging.info('··· Total number of previously identified invariant and almost invariant features = % d ···' %len(features_to_remove))

    if correlated_features == True:
        logging.info('Finding the highly correlated features...')
        # Identify the highly correlated features
        corr_features = get_correlated_features(processed_data, feature_cols)
        logging.info('··· Total number of highly correlated features = % d ···' %len(corr_features))
        features_to_remove.update(corr_features)

    if invariant_features == True:
        logging.info('Finding the invariant features...')
        # Identify the invariant features
        inv_features = get_invariant_features(processed_data, feature_cols)
        logging.info('··· Total number of invariant features = % d ···' %len(inv_features))
        features_to_remove.update(inv_features)    

    # Return the features to be removed
    logging.info('····················································')
    logging.info('··· Total number of features to be removed = % d ···' %len(features_to_remove))
    return features_to_remove  

###############################################################################
# ====================== Molecular Descriptors Calculation ====================
###############################################################################

def get_Mol_objects(inchi_smiles_data):
    # Create an empty list to store all Mol objects
    mol_objects = []

    # Iterate through all compounds 
    for index, row in inchi_smiles_data.iterrows():
        # Get compound InChiKey and SMILES
        cpd_inchi = row['CPD_INCHIKEY']
        cpd_smiles = row['CPD_STD_SMILES']
        # Construct a molecule from the SMILES string
        mol = Chem.rdmolfiles.MolFromSmiles(cpd_smiles)
        if mol == None:
            # Print a warning message
            logging.warning('It was not possible to construct a molecule for the compound with InChiKey' %cpd_inchi)
        # Append the built Mol object to the list
        mol_objects.append(mol)

    # Return the resulting list of Mol objects
    return mol_objects  

def select_descriptors(descriptors, invariant_threshold, correlation_threshold, nan_descriptors=False, invariant_descriptors=True, correlated_descriptors=True):

    # Create an empty set to store the descriptors to be removed
    descriptors_to_remove = set()
    
    if nan_descriptors == True:
        logging.info("Finding the null descriptors...")

        # Create an empty set to store the null descriptors 
        nan_desc = set()
        # Iterate over all descriptors to identify those containing NaN values
        for desc in descriptors.iloc[:,1:].columns.tolist():
            if descriptors[desc].isnull().values.any():
                nan_desc.add(desc)
        logging.info("··· Total number of null descriptors = %d ···" %len(nan_desc))

        # Remove the null descriptors from the descriptor dataset
        descriptors = descriptors.drop(nan_desc, axis=1)
        
    if invariant_descriptors == True:
        logging.info("Finding the invariant and almost invariant descriptors...")
        
        # Create an empty set to store the invariant descriptors
        invariant_desc = set()
        # Iterate over all descriptors to identify those with the same value for all molecules
        for desc in descriptors.iloc[:,1:].columns.tolist():
            # Count the number of descriptor unique values
            unique_values = descriptors[desc].nunique(dropna=True)
            if unique_values == 1:
                invariant_desc.add(desc)
        logging.info("··· Total number of invariant descriptors = %d ···" %len(invariant_desc))

        # Create an empty set to store the almost invariant descriptors
        almost_invariant_desc = set()
        # Iterate over all descriptors to identify those with the mode in more than the defined threshold
        for desc in descriptors.iloc[:, 1:].columns.tolist():
            # Count the occurrences of every unique value
            value_counts = descriptors[desc].value_counts(dropna=True)
            # Calculate the percentage of the most common value (i.e., the mode)
            percent = value_counts.iloc[0] / descriptors.shape[0]
            if percent > invariant_threshold:
                almost_invariant_desc.add(desc)
        logging.info("··· Total number of almost invariant descriptors = %d ···" %len(almost_invariant_desc))

        # Add the identified invariant and almost invariant descriptors to the descriptor set
        descriptors_to_remove.update(invariant_desc)
        descriptors_to_remove.update(almost_invariant_desc)

    if correlated_descriptors == True:
        logging.info("Finding the highly correlated descriptors...")

        # Create an empty set to store the correlated descriptors
        correlated_desc = set()

        # Compute the Pearson correlation coefficient for every pair of descriptors
        corr_matrix = descriptors.iloc[:, 1:].corr(method='pearson', numeric_only=True)
        # Get the indices of the upper triangular part of the matrix and select the corresponding values
        indices = np.triu_indices(n=corr_matrix.shape[0], m=corr_matrix.shape[1], k=1)
        corr_values = corr_matrix.values[indices]

        # Plot an histogram to show the distribution of correlation coefficients
        plt.figure(figsize=(10,6))
        plt.hist(corr_values, bins=40, color='skyblue', edgecolor='black') # set the number of bins to 40
        plt.axvline(x=correlation_threshold, color='red', linestyle='--', linewidth=1) # positive threshold
        plt.axvline(x=-correlation_threshold, color='red', linestyle='--', linewidth=1) # negative threshold
        plt.xlabel('Pearson Correlation Coefficient (PCC)')
        plt.ylabel('Frequency')
        plt.title('PCC distribution of pairs of molecular descriptors')
        plt.show()

        # Create an empty set to store the pairs of correlated descriptors
        corr_pairs = set()
        # Set the (absolute) threshold 
        threshold = correlation_threshold
        # Identify pairs of highly correlated descriptors
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]): # iterate only through the upper triangular part
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    pair = (corr_matrix.index[i], corr_matrix.columns[j])
                    corr_pairs.add(pair)

        # Identify the descriptor with largest mean absolute correlation with all descriptors for each pair 
        for pair in corr_pairs:
            mean_corr1 = corr_matrix[pair[0]].abs().mean(numeric_only=True)
            mean_corr2 = corr_matrix[pair[1]].abs().mean(numeric_only=True)
            if mean_corr1 > mean_corr2:
                correlated_desc.add(pair[0])
            else:
                correlated_desc.add(pair[1])
        logging.info("··· Total number of correlated descriptors = %d ···" %len(correlated_desc))

        # Add the identified correlated descriptors to the descriptor set
        descriptors_to_remove.update(correlated_desc)

    # Remove all identified descriptors
    logging.info('·······················································')
    logging.info('··· Total number of descriptors to be removed = % d ···' %len(descriptors_to_remove))
    descriptor_selection_df = descriptors.drop(descriptors_to_remove, axis=1)

    # Return the resulting dataframe
    return descriptor_selection_df

def get_RDKit_1D_descriptors(inchi_smiles_data, missingVal=None):
    ''' 
    Function that calculates the full list of 1D descriptors for a set of molecules. 
    missingVal argument is used if the descriptor cannot be calculated.
    '''
    # Construct the Mol objects from SMILES 
    logging.info('Constructing the RDKit Mol objects of all compounds...')
    mol_objects = get_Mol_objects(inchi_smiles_data)
    logging.info("··· Total number of Mol objects: %d ···" %len(mol_objects))

    # Create an empty list to store the descriptor calculations
    rdkit_descriptors = []
    
    # Iterate over all compounds 
    logging.info('Computing the RDKit 1D descriptors for every compound...')
    for mol in mol_objects:
        # Create an empty dictionary to store the descriptor calculations
        mol_descriptors = {}
        # Iterate over all descriptor functions 
        for function_name, descriptor_function in Descriptors._descList:
            # Calculate the descriptor value
            try:
                value = descriptor_function(mol)
            # Catch the errors throwed by some descriptor fucntions if they fail
            except Exception as e:
                # Print the error message
                logging.error(str(e))
                # Set the descriptor value to missingVal
                value = missingVal
            mol_descriptors[function_name] = value
        # Append the resulting dictionary to the list
        rdkit_descriptors.append(mol_descriptors)
    
    # Convert the list into a Pandas dataframe and add compound InChiKey'set
    logging.info('Creating the final dataframe...')
    rdkit_descriptors_df = pd.DataFrame(rdkit_descriptors)
    rdkit_descriptors_df.columns = rdkit_descriptors_df.columns.map(lambda x: 'desc_' + str(x)) # rename the column names
    logging.info("··· Total number of 1D descriptors = %d x %d compounds ···" %(rdkit_descriptors_df.shape[1], rdkit_descriptors_df.shape[0]))

    # Add compound InChiKey's to the created dataframe
    rdkit_descriptors_df.insert(0, 'CPD_INCHIKEY', inchi_smiles_data['CPD_INCHIKEY'])
    
    # Return the resulting dataframe
    return rdkit_descriptors_df

def get_ECFP4_fingerprints(inchi_smiles_data, radius, fpSize):
    # Construct the Mol objects from SMILES 
    logging.info('Constructing the RDKit Mol objects of all compounds...')
    mol_objects = get_Mol_objects(inchi_smiles_data)
    logging.info("··· Total number of Mol objects: %d ···" %len(mol_objects))

    # Create a generator for ECFP4 fingerprints 
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)

    # Compute the ECFP4 fingerprint for each compound and get them as a numpy array
    logging.info('Computing the ECFP4 descriptor for every compound...')
    ecfp4_fp = np.array([fpgen.GetFingerprintAsNumPy(mol) for mol in mol_objects])

    # Create a dataFrame for ECFP4 fingerprints
    logging.info('Creating the final dataframe...')
    ecfp4_fp_df = pd.DataFrame(data=ecfp4_fp, columns=[f'ecfp4_{i+1}' for i in range(ecfp4_fp.shape[1])])

    # Add compound InChiKey's to the created dataframe
    ecfp4_fp_df.insert(0, 'CPD_INCHIKEY', inchi_smiles_data['CPD_INCHIKEY'])

    # Return the resulting dataframe
    return ecfp4_fp_df

def get_Mordred_descriptors(inchi_smiles_data):
    # Create a generator for Mordred descriptors 
    transformer = MoleculeTransformer(featurizer='mordred', dtype=float)

    # Compute the Mordred descriptors for all compounds 
    logging.info('Computing the Mordred descriptors for every compound...')
    mordred_desc = np.array([transformer(smiles) for smiles in inchi_smiles_data['CPD_STD_SMILES']])
    mordred_desc = mordred_desc.reshape((mordred_desc.shape[0], -1)) # reshape the array to be 2-dimensional

    # Create a dataFrame for Mordred descriptors
    logging.info('Creating the final dataframe...')
    mordred_df = pd.DataFrame(data=mordred_desc, columns=[f'mordred_{i+1}' for i in range(mordred_desc.shape[1])])

    # Add compound InChiKey's to the created dataframe
    mordred_df.insert(0, 'CPD_INCHIKEY', inchi_smiles_data['CPD_INCHIKEY'])

    # Return the resulting dataframe
    return mordred_df

def compute_molecular_descriptors(data, descriptor_type, radius=2, fpSize=2048, descriptor_selection=True):

    # Define the function arguments and their expected data types and allowed values
    args_type = {'descriptor_type': str, 'radius': int, 'fpSize': int, 'descriptor_selection': bool}
    
    args_values = {'descriptor_type': ['1D_RDKit', 'ECFP4', 'Mordred']}
    
    # Validate argument data types and allowed values
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value.")
        # Check allowed values for specific arguments
        if argument in args_values:
            value = locals()[argument]
            if value not in args_values[argument]:
                raise ValueError(f"'{argument}' must be one of {args_values[argument]} values.")

    # Load the Cell Painting data with chemical annotations
    logging.info('Loading the input data...')
    inchi_smiles_df = pd.read_csv('data/'+data, usecols=['CPD_INCHIKEY', 'CPD_STD_SMILES'], low_memory=False)
    
    if descriptor_type == '1D_RDKit':
        # Compute the RDKit 1D descriptors
        logging.info('### RDKit 1D molecular descriptors ###')
        cpd_descriptors = get_RDKit_1D_descriptors(inchi_smiles_df)
        # Perform descriptor selection
        logging.info('# Descriptor selection #')
        if descriptor_selection == True:
            cpd_descriptors = select_descriptors(cpd_descriptors, invariant_threshold=0.95, correlation_threshold=0.9, nan_descriptors=False)

    elif descriptor_type == 'ECFP4':
        # Compute the ECFP4 fingerprints
        logging.info('### ECFP4 fingerprints ###')
        cpd_descriptors = get_ECFP4_fingerprints(inchi_smiles_df, radius, fpSize)
        # Perform descriptor selection
        logging.info('# Descriptor selection #')
        if descriptor_selection == True:
            if fpSize == 2048:
                cpd_descriptors = select_descriptors(cpd_descriptors, invariant_threshold=0.99, correlation_threshold=0.8, nan_descriptors=False)
            if fpSize == 1024:
                cpd_descriptors = select_descriptors(cpd_descriptors, invariant_threshold=0.97, correlation_threshold=0.8, nan_descriptors=False)

    elif descriptor_type == 'Mordred':
        # Compute the Mordred descriptors
        logging.info('### Mordred descriptors ###')
        cpd_descriptors = get_Mordred_descriptors(inchi_smiles_df)
        # Perform descriptor selection
        logging.info('# Descriptor selection #')
        if descriptor_selection == True:
            cpd_descriptors = select_descriptors(cpd_descriptors, invariant_threshold=0.95, correlation_threshold=0.9, nan_descriptors=True)

    # Return the computed descriptors
    return cpd_descriptors

###############################################################################
# ========== Cell Painting Morphological Features Predictive Models ==========
###############################################################################

random_seed = 42

def cp_create_complete_dataset(cellpainting_data, descriptors):
    # Load the Cell Painting data with chemical annotations
    logging.info('Loading the Cell Painting data...')
    cellpaint_df = pd.read_csv('data/'+cellpainting_data, low_memory=False)

    # Drop invariant, almost invariant and discrete features
    logging.info('Droping the invariant, almost invariant and discrete morphological features...')
    cellpaint_df = cellpaint_df.loc[:, ~((cellpaint_df.columns.str.endswith('Count')) | (cellpaint_df.columns.str.endswith('EulerNumber')) | (cellpaint_df.columns.str.contains('Correlation_Manders')) | (cellpaint_df.columns.str.contains('NumberOfNeighbors')))]

    # Load the molecular descriptors
    logging.info('Loading the molecular descriptors...')
    moldesc_df = pd.read_csv('data/'+descriptors, sep='\t') 

    # Merge both dataframes
    logging.info('Merging the two dataframes...')
    merged_df = pd.merge(cellpaint_df, moldesc_df, on='CPD_INCHIKEY')

    return merged_df

def cp_data_standardisation(X_train, X_test):
    # Define the scaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Convert the scaled arrays back to dataframes with the original index
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    # Return the scaled dataframes
    return X_train, X_test

def plot_flaml_history(time_history, best_valid_loss_history):
    plt.title("Learning Curve")
    plt.xlabel("Wall Clock Time (s)")
    plt.ylabel("Validation Accuracy")
    plt.step(time_history, 1 - np.array(best_valid_loss_history), where="post")
    plt.show()

def cp_evaluate_model_performance(y_true, y_pred):
    # Compute several evaluation metrics for regression
    r2 = metrics.r2_score(y_true, y_pred)
    meanAbsErr = metrics.mean_absolute_error(y_true, y_pred)
    meanSqErr = metrics.mean_squared_error(y_true, y_pred)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_true, y_pred)) 

    # Return the calculated metrics
    return r2, meanAbsErr, meanSqErr, rootMeanSqErr

def plot_actual_vs_predicted(train_true, train_pred, test_true, test_pred):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for the training set
    ax1.scatter(train_true, train_pred, alpha=0.5, s=10)
    ax1.set_xlim(left=-5, right=6)
    ax1.set_ylim(bottom=-2.5, top=2.5)
    ax1.set_title('Training Set: Actual vs. Predicted Values')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')

    # Plot for the test set
    ax2.scatter(test_true, test_pred, alpha=0.5, s=10)
    ax2.set_xlim(left=-5, right=6)
    ax2.set_ylim(bottom=-2.5, top=2.5)
    ax2.set_title('Test Set: Actual vs. Predicted Values')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

def cp_plot_model_feature_importance(CP_feature, model, X_train, top_n=10):
    feature_importance = model.feature_importances_
    feature_names = X_train.columns.tolist()

    # Get the indices that would sort the feature importance array in descending order
    sorted_idx = np.argsort(feature_importance)[::-1]

    # Plot feature importance with sorted bars
    plt.figure(figsize=(10,6))
    plt.suptitle("Overall Feature Importance", fontsize=16)

    sns.barplot(x=feature_importance[sorted_idx[:top_n]], y=np.array(feature_names)[sorted_idx[:top_n]], palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f'{CP_feature} morphological feature', fontsize=16)
    plt.show()

def cp_model_training_and_evaluation(data, CP_feature, train_split=0.8, verbose_flaml=False, plot_results=False, plot_feature_importance=False, save_results=True, results_filename=None):

    descriptors_dict = {'desc':'RDKit 1D descriptors', 'ecfp4':'ECFP4 fingerprints', 'mordred':'Mordred descriptors', 'pc': 'Physicochemical properties'}

    # Define the function arguments and their expected data types and allowed values
    args_type = {'train_split': float, 'verbose_flaml': bool, 'plot_results': bool, 'plot_feature_importance': bool, 'save_results': bool, 'results_filename': Optional[str]}
    
    CP_feature_values = [col_name[3:] for col_name in data.columns.tolist() if col_name.startswith('cp_')] + ['all']

    # Validate argument data types
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value.")
    
    # Validate CP_feature argument
    if isinstance(CP_feature, str):
        CP_feature = [CP_feature] 
    elif not isinstance(CP_feature, list):
        raise TypeError("'CP_feature' must be either a string or a list of strings.")
    
    # Check allowed values for the CP_feature argument
    for feature in CP_feature:
        if feature not in CP_feature_values:
            raise ValueError(f"'{feature}' is not a valid value for 'CP_feature'. It must be one of {CP_feature_values} values.")
            
    # Check for NaN in any row
    logging.info('Checking for NaN values in any row...')
    nan_rows = []
    for index, row in data.iterrows():
        if row.isnull().any():
            nan_rows.append(index)
    logging.info('··· Compounds with one or more NaN values: %d ···' %len(nan_rows))
    # Remove the rows containing NaN values 
    data = data.drop(nan_rows)
    
    # Define the prefix set for X and the prefix for Y
    x_prefixes = {'ecfp4', 'desc', 'mordred', 'pc'} 
    y_prefix = 'cp'

    # Determine the X prefix 
    col_prefixes = set([col_name.split('_')[0] for col_name in data.columns.tolist()])
    x_prefix = next(iter(x_prefixes & col_prefixes))

    # Split the dataset into training and test sets
    logging.info('Splitting the dataset into training and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(data.filter(regex=f'^{x_prefix}', axis=1), data.filter(regex=f'^{y_prefix}', axis=1), train_size=train_split, random_state=random_seed)
    logging.info(f'Size of X_train {X_train.shape} and size of y_train {y_train.shape}')
    logging.info(f'Size of X_test {X_test.shape} and size of y_test {y_test.shape}')

    if x_prefix != 'ecfp4':
        # Standardise the data
        logging.info('Standardising the descriptor data...')
        X_train, X_test = cp_data_standardisation(X_train, X_test)

    # Create the list of morphological feature to be modeled
    if len(CP_feature) == 1 and CP_feature[0] == 'all':
        y_features = [feat for feat in y_train.columns.tolist()]
    else:
        y_features = ['cp_'+feature for feature in CP_feature]

    # Set the list of time budgets
    time_budget = 60

    # Create an empty list to store metrics dictionaries
    metrics_list = []

    # Change the current working direcotry
    original_directory = os.getcwd()
    new_directory = 'log/'+x_prefix+'_logs'

    if not os.path.exists(new_directory):
        logging.info('Creating a new directory to save the AutoML log files...')
        os.makedirs(new_directory)
    os.chdir(new_directory)

    # Iterate over all Cell Painting features
    for feature in y_train.columns.tolist():

        # Check if the feture has to be modeled
        if feature in y_features:

            # Select the Cell Painting feature in both training and test sets
            y_train_select = y_train.loc[:, feature]
            y_test_select = y_test.loc[:, feature]

            logging.info(f'\n\n·················{feature[3:]} ({y_train.columns.tolist().index(feature)+1}/{len(y_train.columns)})·················\n')

            # Create an AutoML class for tuning the hyperparameters and select the best model
            automl = AutoML()
            automl.fit(X_train=X_train, y_train=y_train_select, 
                       task="regression", estimator_list='auto', metric='mse', eval_method='auto', 
                       time_budget=time_budget, early_stop=False,
                       verbose=-1, n_jobs=-1, log_type='all', log_file_name=f'{feature[3:]}.log', seed=random_seed)

            logging.info(f'Best estimator: {automl.best_estimator}')
            logging.info(f'Best hyperparameter config: {automl.best_config}')
            logging.info(f'Best MSE on validation data: {1 - automl.best_loss:.4g}')
            logging.info(f'Training duration of best run: {automl.best_config_train_time:.4g} s')

            if verbose_flaml == True:
                # Plot the FLAML history
                time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = get_output_from_log(filename=f'{feature[3:]}.log', time_budget=time_budget)
                plot_flaml_history(time_history, best_valid_loss_history)

            # Get predictions on the test set
            y_test_pred = automl.predict(X_test)
            # Evaluate the model performance on the test set
            r2_test, meanAbsErr_test, meanSqErr_test, rootMeanSqErr_test = cp_evaluate_model_performance(y_test_select, y_test_pred)

            # Get predictions on the training set
            y_train_pred = automl.predict(X_train)
            # Evaluate the model performance on the test set
            r2_train, meanAbsErr_train, meanSqErr_train, rootMeanSqErr_train = cp_evaluate_model_performance(y_train_select, y_train_pred)

            logging.info(f'R2 (train) = {r2_train} --- MAE (train) = {meanAbsErr_train} --- MSE (train) = {meanSqErr_train} --- RMSE (train) = {rootMeanSqErr_train}')
            logging.info(f'R2 (test) = {r2_test} --- MAE (test) = {meanAbsErr_test} --- MSE (test) = {meanSqErr_test} --- RMSE (test) = {rootMeanSqErr_test}')

            # Add the computed evaluation metrics to the dataframe
            metrics_list.append({'descriptor':descriptors_dict[x_prefix], 'CP_feature':feature[3:], 'time_budget': time_budget, 'ML_model':automl.best_estimator, 
                                 'r2_score(train)':r2_train, 'r2_score(test)':r2_test, 'MAE(train)':meanAbsErr_train, 'MAE(test)':meanAbsErr_test,
                                 'MSE(train)':meanSqErr_train, 'MSE(test)':meanSqErr_test, 'RMSE(train)':rootMeanSqErr_train, 'RMSE(test)':rootMeanSqErr_test})

            if plot_results == True:
                # Plot the actual vs. predicted values
                plot_actual_vs_predicted(y_train_select, y_train_pred, y_test_select, y_test_pred)
            
            if hasattr(automl.model.estimator, 'feature_importances_') and plot_feature_importance == True:
                cp_plot_model_feature_importance(feature[3:], automl.model.estimator, X_train)

    # Change the current working directory
    os.chdir(original_directory)

    # Create the DataFrame containing the metrics data
    metrics_df = pd.DataFrame(metrics_list)

    # Save the results in a TSV file
    if save_results == True:
        if results_filename.endswith('tsv'):
            metrics_df.to_csv('results/'+results_filename, sep='\t', index=False)
        elif results_filename.endswith('csv'):
            metrics_df.to_csv('results/'+results_filename, index=False)

    # Return the metrics dataframe
    return metrics_df 

###############################################################################
# ======================= Pharmacology Predictive Models ======================
###############################################################################

def create_complete_dataset(input_data, pharmacology_safety_data):
    # Load the input morphological or chemical data with compound annotations
    logging.info('Loading the input data...')
    if input_data.endswith('.csv'):
        input_df = pd.read_csv('data/'+input_data, low_memory=False)
    elif input_data.endswith('.tsv'):
        input_df = pd.read_csv('data/'+input_data, sep='\t')

    # Load the pharmacology/safety data with compound InChiKey identifier
    logging.info('Loading the pharmacology/safety binary matrix...')
    pharmacology_df = pd.read_csv('data/'+pharmacology_safety_data) 

    # Construct the final dataframe 
    logging.info('Merging the two dataframes...')
    merged_df = pd.merge(input_df, pharmacology_df, on='CPD_INCHIKEY')

    return merged_df

def data_standardisation(X_train, X_test):
    # Define the columns to standardise
    columns_to_standardise = X_train.columns[X_train.columns.str.startswith(('desc', 'mordred', 'pc'))]

    # Define the scaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = X_train.copy()
    if len(columns_to_standardise) != 0:
        X_train_scaled[columns_to_standardise] = scaler.fit_transform(X_train[columns_to_standardise])

    # Transform the test data using the same scaler
    X_test_scaled = X_test.copy()
    if len(columns_to_standardise) != 0:
        X_test_scaled[columns_to_standardise]= scaler.transform(X_test[columns_to_standardise])

    # Return the scaled dataframes
    return X_train_scaled, X_test_scaled

def plot_target_distribution(zero_counts, one_counts, target_id, show_plot=False, save_path=None):
    # Create a pie chart
    labels = ['0', '1']
    sizes = [zero_counts, one_counts]
    colors = ['lightcoral', 'lightskyblue']
    
    plt.figure(figsize=(10,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title(f'{target_id} target --- Class Distribution')

    if save_path:
        plt.savefig(save_path+'/target_distribution_plot.png', format='png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def objective(trial, algorithm_name, X_train_data, y_train_data):

    if algorithm_name == 'GradientBoosting':
        # Define the grid of hyperparameters to be optimized
        params = {
            "criterion": "friedman_mse",
            "loss": "log_loss",
            "n_estimators": trial.suggest_int("n_estimators", 10, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2', None]),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.05, 1),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.001, 0.5),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.001, 0.5),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 1.0),
            "verbose": 0
        }
        # Define the Gradient Boosting Classifier model
        model = GradientBoostingClassifier(**params, random_state=42)

    elif algorithm_name == 'XGBoost':
        # Define the grid of hyperparameters to be optimized
        params = {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "tree_metod": "exact",
            "n_estimators": trial.suggest_int("n_estimators", 10, 300),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.01, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "subsample": trial.suggest_float("subsample", 0.05, 0.5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "verbosity": 0
        }
        # Define the XGBoost Classifier model
        model = XGBClassifier(**params, random_state=42) 

    # Define the model evaluation method
    n_folds = 5
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Create two empty arrays to store both validation and training scores
    val_scores = np.empty(n_folds)
    train_scores = np.empty(n_folds)

    y_train_data = y_train_data.to_frame() # convert the Series to DataFrame
    for idx, (train_idx, valid_idx) in enumerate(cv.split(X_train_data, y_train_data)):
        X_train, X_valid = X_train_data.iloc[train_idx, :], X_train_data.iloc[valid_idx, :]
        y_train, y_valid = y_train_data.iloc[train_idx, :].values.ravel(), y_train_data.iloc[valid_idx, :].values.ravel()

        # Fit the model
        model.fit(X_train, y_train, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train))
  
        # Evaluate the fold on both validation and training sets
        val_scores[idx] = metrics.f1_score(y_valid, model.predict(X_valid))
        train_scores[idx] = metrics.f1_score(y_train, model.predict(X_train))

        # Check the custom pruning condition
        if val_scores[idx] < 0.1: 
            # Prune the trial if the F1-score for the validation set is lower than 0.1
            trial.report(val_scores[idx], step=idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # Set the mean training score as a user attribute
    trial.set_user_attr('train_score', np.mean(train_scores))

    # Return the mean of the validation scores 
    return np.mean(val_scores)

def plot_train_vs_validation_loss(training_scores, validation_scores, best_trial, show_plot=False, save_path=None):
    plt.figure(figsize=(10,6))
    plt.plot(training_scores, marker='o', markersize=2, label='Mean Training CV Score')
    plt.plot(validation_scores, marker='', markersize=2, label='Mean Validation CV Scores')
    plt.scatter([best_trial], [training_scores[best_trial]], color='r', marker='o', s=5, label='Best Trial (Training)')
    plt.scatter([best_trial], [validation_scores[best_trial]], color='r', marker='o', s=5, label='Best Trial (Validation)')
    plt.xlabel('Trial')
    plt.ylabel('Score')
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+'/training_vs_validation_loss_plot.png', format='png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def evaluate_model_performance(y_true, y_pred):
    try: 
        # Compute several evaluation metrics for classification
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        avg_precision = metrics.average_precision_score(y_true, y_pred)

        # Return the calculated metrics
        return tn, fp, fn, tp, balanced_accuracy, precision, recall, f1, roc_auc, mcc, avg_precision
    
    except ValueError as e:
        logging.debug(f'Error: {e}')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
def plot_model_evaluation(target_id, algorithm_name, y_true, y_pred_proba, y_pred_class, show_plot=False, save_path=None, training_data=False):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
    fig.suptitle(f'{target_id} endpoint --- {algorithm_name} model', fontsize=16)

    # Confusion Matrix
    plot_confusion_matrix(y_true, y_pred_class, ax=axs[0])
    axs[0].set_title("Confusion Matrix")

    # ROC Curve
    plot_roc(y_true, y_pred_proba, ax=axs[1])
    axs[1].set_title("ROC Curve")

    # Precision-Recall Curve
    plot_precision_recall(y_true, y_pred_proba, ax=axs[2])
    axs[2].set_title("Precision-Recall (PR) Curve")

    plt.tight_layout()

    if save_path:
        if training_data:
            plt.savefig(save_path+'/model_evaluation_plots_training_set.png', format='png')
        else:
            plt.savefig(save_path+'/model_evaluation_plots.png', format='png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_model_feature_importance(target_id, algorithm_name, model, X_train, top_n=10, show_plot=False, save_path=None):
    feature_importance = model.feature_importances_
    feature_names = X_train.columns.tolist()

    # Get the indices that would sort the feature importance array in descending order
    sorted_idx = np.argsort(feature_importance)[::-1]

    # Plot feature importance with sorted bars
    plt.figure(figsize=(10,6))
    plt.suptitle("Overall Feature Importance", fontsize=16)

    sns.barplot(x=feature_importance[sorted_idx[:top_n]], y=np.array(feature_names)[sorted_idx[:top_n]], palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f'{target_id} endpoint --- {algorithm_name} model', fontsize=16)

    if save_path:
        plt.savefig(save_path+'/model_feature_importance_plot.png', format='png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def tg_model_training_and_evaluation(data, target, check_target_distribution=False, train_split=0.8, optuna_trials=200, verbose_optuna=False, plot_loss_model=False, plot_results=False, plot_feature_importance=False, results_filename=None):
    
    data_dict = {'cp':'Cell Painting', 'desc':'RDKit 1D descriptors', 'ecfp4':'ECFP4 fingerprints', 'mordred':'Mordred descriptors', 'pc': 'Physicochemical properties'}

    # Define the function arguments and their expected data types and allowed values
    args_type = {'check_target_distribution': bool, 'train_split': float, 'optuna_trials': int, 'verbose_optuna': bool, 'plot_loss_model': bool, 'plot_results': bool, 'plot_feature_importance': bool, 'results_filename': Optional[str]}
    
    target_values = [col_name[3:] for col_name in data.columns.tolist() if col_name.startswith('tg_')] + ['all']

    # Validate argument data types
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value.")
    
    # Validate 'target' argument
    if isinstance(target, str):
        target = [target] 
    elif not isinstance(target, list):
        raise TypeError("'target' must be either a string or a list of strings.")
    
    # Check allowed values for the 'target' argument
    for tg in target:
        if tg not in target_values:
            raise ValueError(f"'{tg}' is not a valid value for 'target'. It must be one of {target_values} values.")
    
    # Define the prefix set for X and the prefix for Y
    x_prefixes = {'cp', 'ecfp4', 'desc', 'mordred', 'pc'} 
    y_prefix = 'tg'

    # Determine the X prefix 
    col_prefixes = set([col_name.split('_')[0] for col_name in data.columns.tolist()])
    x_prefix = '|'.join(x_prefixes & col_prefixes)

    # Define X and y
    X_data = data.filter(regex=f'^{x_prefix}', axis=1)
    y_data = data.filter(regex=f'^{y_prefix}', axis=1)

    # Define the list of algorithms and set the number of trials
    algorithms_list = ['GradientBoosting', 'XGBoost']
    num_trials = optuna_trials

    # Create the list of targets to be modeled
    if len(target) == 1:
        if target[0] == 'all':
            y_features = [tg for tg in y_data.columns.tolist()]
        else:
            y_features = ['tg_'+target[0]]
    else:
        y_features = ['tg_'+tg for tg in target]

    # Create the descriptor directory
    input_data_directory = 'log/'+'_'.join(list(x_prefixes & col_prefixes))+'_logs'
    if not os.path.exists(input_data_directory):
        logging.info('Creating a new directory to save Optuna results and additional plots...')
        os.makedirs(input_data_directory)

    # Write the column names in the results file
    if not os.path.exists('results/'+results_filename):
        with open('results/'+results_filename, 'a') as resultsfile:
            resultsfile.write('data\ttarget\tpositive_counts\tnegative_counts\tA:I_ratio\tML_model\tnum_trials\tbest_trial\t'
                            'tp(train)\tfp(train)\tfn(train)\ttn(train)\ttp\tfp\tfn\ttn\tbalanced_accuracy(train)\tbalanced_accuracy\tprecision(train)\tprecision\trecall(train)\trecall\t'
                            'f1_score(train)\tf1_score(validation)\tf1_score\troc_auc_score(train)\troc_auc_score\tmcc(train)\tmcc\tpr_auc_score(train)\tpr_auc_score\ttime\n')

    # Initialize a set to store the combinations of target-algorithm already present in the results file
    existing_combinations = set()
    # Read the TSV file and parse combinations
    with open('results/'+results_filename, 'r') as resultsfile:
        for line in resultsfile:
            columns = line.split('\t')
            target = columns[1]
            algorithm = columns[5]
            existing_combinations.add((target, algorithm))

    # Iterate over all pharmacological targets 
    for pharmaco_target in y_data.columns.tolist():

        # Check if the target has to be modeled
        if pharmaco_target in y_features:

            # Create the target directory
            target_directory = input_data_directory+'/'+pharmaco_target[3:]
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            logging.info(f'\n\n·················{pharmaco_target[3:]} ({y_data.columns.tolist().index(pharmaco_target)+1}/{len(y_data.columns)})·················\n')

            # Select the target
            y_target_select = y_data.loc[:, pharmaco_target] 

            # Count the instances of each output class
            positive_counts = (y_target_select == 1).sum()
            negative_counts = (y_target_select == 0).sum()
            logging.info(f'Positive counts: {positive_counts}')
            logging.info(f'Negative counts: {negative_counts}')
            logging.info(f'Ratio: {positive_counts / negative_counts}')

            # Plot the target values distribution
            plot_target_distribution(negative_counts, positive_counts, pharmaco_target[3:], show_plot=check_target_distribution, save_path=target_directory)

            # Down-sample the dataset if the ratio is below 1:10
            if positive_counts / negative_counts < 0.1:
                logging.info('Down-sampling the dataset to get a 1:10 ratio...')
                # Select the indices of all rows with 1
                positive_indices = y_target_select.loc[y_target_select == 1].index.tolist()
                # Randomly select indices of rows with 0 to get 1:10 ratio
                target_negative_count = int(positive_counts/0.1) #- positive_counts
                negative_indices = y_target_select.loc[y_target_select == 0].sample(n=target_negative_count, random_state=42).index.tolist()
                # Concatenate the selected indices
                selected_indices = positive_indices + negative_indices                
                # Get the selected rows
                X_data_selection = X_data.loc[selected_indices]
                y_target_select = y_target_select.loc[selected_indices]
            else:
                X_data_selection = X_data.copy()

            # Split the dataset into training and test sets
            logging.info('Splitting the dataset into training and test sets...')
            X_train, X_test, y_train_select, y_test_select = train_test_split(X_data_selection, y_target_select, train_size=train_split, random_state=42)
            logging.info(f'Size of X_train {X_train.shape} and size of y_train {y_train_select.shape}')
            logging.info(f'Size of X_test {X_test.shape} and size of y_test {y_test_select.shape}')

            # Standardise the data
            logging.info('Standardising the descriptor data...')
            X_train, X_test = data_standardisation(X_train, X_test)

            for algorithm in algorithms_list:

                # Create the algorithm directory
                algorithm_directory = target_directory+'/'+algorithm
                if not os.path.exists(algorithm_directory):
                    os.makedirs(algorithm_directory)
                
                logging.info(f'\n=================={algorithm}===================')
                start_time = time.time()

                # Check if the combination target-algorithm already exists in the results file
                if (pharmaco_target[3:], algorithm) in existing_combinations:
                    continue

                # Create an Optuna study to perform the hyperparameter optimization for the corresponding algorithm
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda trial: objective(trial, algorithm, X_train, y_train_select), n_trials=num_trials) # n_jobs=18

                # Save the study object to a file
                with open(algorithm_directory+'/optuna_study.pickle', 'wb') as f:
                    pickle.dump(study, f)

                # Save the Optuna results 
                trials_df = study.trials_dataframe()
                trials_df.to_csv(algorithm_directory+'/optuna_trials.tsv', sep='\t', index=False)

                # Retrieve both training and validation scores for each trial
                train_scores = [trial.user_attrs['train_score'] if 'train_score' in trial.user_attrs else 0 for trial in study.trials]
                val_scores = [trial.value for trial in study.trials]
                val_scores = [0 if train_score == 0 else val_score for train_score, val_score in zip(train_scores, val_scores)] # set val_score to 0 if the trail has been pruned

                if verbose_optuna == True:
                    # Plot the optimization history
                    optuna.visualization.plot_optimization_history(study).show()
                    # Plot the parallel coordinate plot
                    optuna.visualization.plot_parallel_coordinate(study).show()
                    # Plot the pruning history
                    optuna.visualization.plot_intermediate_values(study).show()

                # Plot the training vs. validation scores curve
                plot_train_vs_validation_loss(train_scores, val_scores, study.best_trial.number, show_plot=plot_loss_model, save_path=algorithm_directory)

                logging.info(f'Best trial: {str(study.best_trial.number)}')
                logging.info(f'Best params: {str(study.best_params)}')

                # Re-define the model with the best hyperparameter combination
                if algorithm == 'GradientBoosting':
                    model = GradientBoostingClassifier(**study.best_params, criterion='friedman_mse', loss='log_loss', verbose=0, random_state=42)
                elif algorithm == 'XGBoost':
                    model = XGBClassifier(**study.best_params, objective='binary:logistic', booster='gbtree', tree_method='exact', verbosity=0, random_state=42)

                # Train the model
                model.fit(X_train, y_train_select, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_select))

                # Save the trained model to a file
                with open(algorithm_directory+'/model.pickle', "wb") as fout:
                    pickle.dump(model, fout)
                
                # Get predictions on the test set
                y_test_pred_proba = model.predict_proba(X_test)
                # Convert them to a DataFrame 
                prob_df = pd.DataFrame(y_test_pred_proba)
                # Extract the compound identifiers and convert them to a DataFrame
                inchis_df = pd.DataFrame({'CPD_INCHIKEY': data.loc[y_test_select.index, 'CPD_INCHIKEY']})
                # Concatenate both DataFrames
                inchis_prob_df = pd.concat([inchis_df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
                # Save it to a CSV file
                inchis_prob_df.to_csv(algorithm_directory+'/test_predicted_probabilities.csv', index=False)

                # Set the threshold and classify the predicted proabilities
                threshold = 0.5
                y_test_pred_class = (y_test_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the test set
                tn_test, fp_test, fn_test, tp_test, balanced_accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, mcc_test, avg_precision_test = evaluate_model_performance(y_test_select, y_test_pred_class)

                # Get predictions on the training set
                y_train_pred_proba = model.predict_proba(X_train)
                # Classify the predicted proabilities
                y_train_pred_class = (y_train_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the training set
                tn_train, fp_train, fn_train, tp_train, balanced_accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, mcc_train, avg_precision_train = evaluate_model_performance(y_train_select, y_train_pred_class)

                end_time = time.time() 
                total_time = (end_time - start_time) / 60

                logging.info(f'Time: {total_time:.2f} minutes')
                logging.info(f'Balanced Accuracy (train) = {balanced_accuracy_train} --- Precision (train) = {precision_train} --- Recall (train)= {recall_train} --- F1-score (train)= {f1_train} --- ROC AUC score (train) = {roc_auc_train} --- MCC (train)= {mcc_train} --- PR AUC score (train)= {avg_precision_train}')
                logging.info(f'Balanced Accuracy (test) = {balanced_accuracy_test} --- Precision (test) = {precision_test} --- Recall (test)= {recall_test} --- F1-score (test)= {f1_test} --- ROC AUC score (test) = {roc_auc_test} --- MCC (test)= {mcc_test} --- PR AUC score (test)= {avg_precision_test}')
                logging.info(f'F1-score (validation): {study.best_trial.value}')

                # Construct the 'data' column 
                data_column = ', '.join(map(str, [data_dict[prefix] for prefix in x_prefixes & col_prefixes]))

                # Write values to the results file
                with open('results/'+results_filename, 'a') as resultsfile:
                    resultsfile.write(f'{data_column}\t{pharmaco_target[3:]}\t{positive_counts} ({round((positive_counts/(positive_counts+negative_counts))*100, 2)}%)\t{negative_counts} ({round((negative_counts/positive_counts+negative_counts)*100, 2)}%)\t{round(positive_counts/negative_counts, 4)}\t'
                                      f'{algorithm}\t{num_trials}\t{study.best_trial.number}\t{tp_train}\t{fp_train}\t{fn_train}\t{tn_train}\t{tp_test}\t{fp_test}\t{fn_test}\t{tn_test}\t'
                                      f'{balanced_accuracy_train}\t{balanced_accuracy_test}\t{precision_train}\t{precision_test}\t{recall_train}\t{recall_test}\t{f1_train}\t{study.best_trial.value}\t{f1_test}\t'
                                      f'{roc_auc_train}\t{roc_auc_test}\t{mcc_train}\t{mcc_test}\t{avg_precision_train}\t{avg_precision_test}\t{total_time}\n')

                # Plot several model evaluation plots
                plot_model_evaluation(pharmaco_target[3:], algorithm, y_train_select, y_train_pred_proba, y_train_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=True)
                plot_model_evaluation(pharmaco_target[3:], algorithm, y_test_select, y_test_pred_proba, y_test_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=False)
            
                if hasattr(model, 'feature_importances_'):
                    # Plot the model feature importances
                    plot_model_feature_importance(pharmaco_target[3:], algorithm, model, X_train, show_plot=plot_feature_importance, save_path=algorithm_directory)

    # Load the results TSV file
    final_df = pd.read_csv('results/'+results_filename, sep='\t')

    # Return the metrics dataframe
    return final_df 

# ======================= Analysis of Individual Models =======================

def get_best_model(results_df, metrics='f1_score(validation)'):
    # Create an empty list to store the index of the best-peforming models
    best_model_indices = []

    # Iterate over targets
    for target in results_df['target'].unique().tolist():
        # Select the models of the given target
        target_results = results_df.loc[results_df['target'] == target]
        
        # Find the best-performing model
        if not target_results.isna().any().any():
            best_model = target_results.loc[[target_results[metrics].idxmax()]]

        # Append the row index to the list
        best_model_indices.append(best_model.index[0])

    # Get the best-performing model for each target
    best_models = results_df.loc[best_model_indices]

    # Return the best-performing models
    return best_models

def get_target_names(results_df):
    # Load the target names
    target_names = pd.read_csv('data/4_data/target_id_to_protein_name.csv')

    # Get the name of the targets
    targets_df = pd.merge(results_df[['target','mcc']], target_names, left_on='target', right_on='target_id')
    targets_df = targets_df.drop('target_id', axis=1) # remove the 'target_id' column
    targets_df.rename(columns={'target':'target_id'}, inplace=True) # rename the 'target' column
    targets_df.rename(columns={'model_name':'target_name'}, inplace=True) # rename the 'model_name' column
    targets_df = targets_df[['target_id','target_name','mcc']]

    return targets_df

def get_target_go_annotations(results_df):
    # Load the target names
    target_names = pd.read_csv('data/4_data/target_id_to_protein_name.csv')

    # Get the name of the targets
    targets_df = pd.merge(results_df[['target','mcc']], target_names, left_on='target', right_on='target_id')
    targets_df = targets_df.drop('target_id', axis=1) # remove the 'target_id' column
    targets_df.rename(columns={'target':'target_id'}, inplace=True) # rename the 'target' column
    targets_df.rename(columns={'model_name':'target_name'}, inplace=True) # rename the 'model_name' column
    targets_df = targets_df[['target_id','target_name','mcc']]

    # Load the GO annotations
    target_go_annotations = pd.read_csv('data/4_data/target_id_to_go_term.csv', dtype=str)

    # Get the GO terms of the modelled targets
    targets_df = pd.merge(targets_df, target_go_annotations[['UniProtID','GO_Term_id','GO_Term_name','GO_Term_namespace',]], left_on='target_id', right_on='UniProtID')
    targets_df = targets_df.drop('UniProtID', axis=1) # remove the 'UniProtID' column

    # Drop the duplicated GO annotations
    targets_df = targets_df.drop_duplicates()

    return targets_df

# ============================= EARLY DATA FUSION =============================

def combine_data_sources(first_dataset, second_dataset, third_dataset=None):
    combined_data = pd.merge(first_dataset, second_dataset, on='CPD_INCHIKEY')
    combined_data = combined_data.loc[:, ~combined_data.columns.str.endswith('_y')] # remove columns ending in '_y'
    combined_data.columns = combined_data.columns.str.replace('_x$', '', regex=True) # remove '_x' from the column names ending in '_x'

    if third_dataset is not None:
        combined_data = pd.merge(combined_data, third_dataset, on='CPD_INCHIKEY')
        combined_data = combined_data.loc[:, ~combined_data.columns.str.endswith('_y')] # remove columns ending in '_y'
        combined_data.columns = combined_data.columns.str.replace('_x$', '', regex=True) # remove '_x' from the column names ending in '_x'

    return combined_data

def perform_feature_selection(X_train, y_train, X_test, k, data_modalities, save_path=None):
    # Define the selector 
    selector = SelectKBest(f_classif, k=k)

    # Fit and transform the training data
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_train_selected = pd.DataFrame(X_train_selected, columns=X_train.columns[selector.get_support(indices=True)])

    # Transform the test data using the same selector
    X_test_selected = selector.transform(X_test) 
    X_test_selected = pd.DataFrame(X_test_selected, columns=X_test.columns[selector.get_support(indices=True)])

    # Count occurrences for each data modality
    modalities_counts = {}
    for modality in data_modalities:
        modalities_counts[modality] = [sum(col.startswith(modality) for col in X_train.columns), sum(col.startswith(modality) for col in X_train_selected.columns)]

    # Save feature counts to a TSV file
    with open(save_path+'/selected_feature_counts.tsv', 'w') as tsvfile:
        tsvfile.write('data_modality\toriginal_counts\toriginal_percentage\tselected_counts\tselected_percentage\n')
        for modality, counts_list in modalities_counts.items():
            tsvfile.write(f"{modality}\t{counts_list[0]}\t{round((counts_list[0]/X_train.shape[1])*100, 2)}\t{counts_list[1]}\t{round((counts_list[1]/X_train_selected.shape[1])*100, 2)}\n")
    
    # Save feature names to a TXT file
    with open(save_path+'/selected_feature_names.txt', 'w') as txtfile:
        for feature in X_train.columns[selector.get_support(indices=True)].tolist():
            txtfile.write(f"{feature}\n")

    # Return the feature selected dataframes
    return X_train_selected, X_test_selected

def tg_early_fusion_model_training_and_evaluation(data, target, k_features=1000, check_target_distribution=False, train_split=0.8, optuna_trials=200, verbose_optuna=False, plot_loss_model=False, plot_results=False, plot_feature_importance=False, results_filename=None):
    
    data_dict = {'cp':'Cell Painting', 'desc':'RDKit 1D descriptors', 'ecfp4':'ECFP4 fingerprints', 'mordred':'Mordred descriptors', 'pc': 'Physicochemical properties'}

    # Define the function arguments and their expected data types and allowed values
    args_type = {'k_features': int, 'check_target_distribution': bool, 'train_split': float, 'optuna_trials': int, 'verbose_optuna': bool, 'plot_loss_model': bool, 'plot_results': bool, 'plot_feature_importance': bool, 'results_filename': Optional[str]}
    
    target_values = [col_name[3:] for col_name in data.columns.tolist() if col_name.startswith('tg_')] + ['all']

    # Validate argument data types
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value.")
    
    # Validate 'target' argument
    if isinstance(target, str):
        target = [target] 
    elif not isinstance(target, list):
        raise TypeError("'target' must be either a string or a list of strings.")
    
    # Check allowed values for the 'target' argument
    for tg in target:
        if tg not in target_values:
            raise ValueError(f"'{tg}' is not a valid value for 'target'. It must be one of {target_values} values.")
        
    # Define the prefix set for X and the prefix for Y
    x_prefixes = {'cp', 'ecfp4', 'desc', 'mordred', 'pc'} 
    y_prefix = 'tg'

    # Determine the X prefix 
    col_prefixes = set([col_name.split('_')[0] for col_name in data.columns.tolist()])
    x_prefix = '|'.join(x_prefixes & col_prefixes)

    # Define X and y
    X_data = data.filter(regex=f'^{x_prefix}', axis=1)
    y_data = data.filter(regex=f'^{y_prefix}', axis=1)

    # Define the list of algorithms and set the number of trials
    algorithms_list = ['GradientBoosting','XGBoost'] 
    num_trials = optuna_trials

    # Create the list of targets to be modeled
    if len(target) == 1:
        if target[0] == 'all':
            y_features = [tg for tg in y_data.columns.tolist()]
        else:
            y_features = ['tg_'+target[0]]
    else:
        y_features = ['tg_'+tg for tg in target]

    # Create the descriptor directory
    input_data_directory = 'log/'+'_'.join(list(x_prefixes & col_prefixes))+'_logs'
    if not os.path.exists(input_data_directory):
        logging.info('Creating a new directory to save Optuna results and additional plots...')
        os.makedirs(input_data_directory)

    # Write the column names in the results file
    if not os.path.exists('results/'+results_filename):
        with open('results/'+results_filename, 'a') as resultsfile:
            resultsfile.write('data\ttarget\tpositive_counts\tnegative_counts\tA:I_ratio\tML_model\tnum_trials\tbest_trial\t'
                            'tp(train)\tfp(train)\tfn(train)\ttn(train)\ttp\tfp\tfn\ttn\tbalanced_accuracy(train)\tbalanced_accuracy\tprecision(train)\tprecision\trecall(train)\trecall\t'
                            'f1_score(train)\tf1_score(validation)\tf1_score\troc_auc_score(train)\troc_auc_score\tmcc(train)\tmcc\tpr_auc_score(train)\tpr_auc_score\ttime\n')

    # Initialize a set to store the combinations of target-algorithm already present in the results file
    existing_combinations = set()
    # Read the TSV file and parse combinations
    with open('results/'+results_filename, 'r') as resultsfile:
        for line in resultsfile:
            columns = line.split('\t')
            target = columns[1]
            algorithm = columns[5]
            existing_combinations.add((target, algorithm))

    # Iterate over all pharmacological targets 
    for pharmaco_target in y_data.columns.tolist():

        # Check if the target has to be modeled
        if pharmaco_target in y_features:

            # Create the target directory
            target_directory = input_data_directory+'/'+pharmaco_target[3:]
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            logging.info(f'\n\n·················{pharmaco_target[3:]} ({y_data.columns.tolist().index(pharmaco_target)+1}/{len(y_data.columns)})·················\n')

            # Select the target
            y_target_select = y_data.loc[:, pharmaco_target] 

            # Count the instances of each output class
            positive_counts = (y_target_select == 1).sum()
            negative_counts = (y_target_select == 0).sum()
            logging.info(f'Positive counts: {positive_counts}')
            logging.info(f'Negative counts: {negative_counts}')
            logging.info(f'Ratio: {positive_counts / negative_counts}')

            # Plot the target values distribution
            plot_target_distribution(negative_counts, positive_counts, pharmaco_target[3:], show_plot=check_target_distribution, save_path=target_directory)

            # Down-sample the dataset if the ratio is below 1:10
            if positive_counts / negative_counts < 0.1:
                logging.info('Down-sampling the dataset to get a 1:10 ratio...')
                # Select the indices of all rows with 1
                positive_indices = y_target_select.loc[y_target_select == 1].index.tolist()
                # Randomly select indices of rows with 0 to get 1:10 ratio
                target_negative_count = int(positive_counts/0.1) #- positive_counts
                negative_indices = y_target_select.loc[y_target_select == 0].sample(n=target_negative_count, random_state=42).index.tolist()
                # Concatenate the selected indices
                selected_indices = positive_indices + negative_indices                
                # Get the selected rows
                X_data_selection = X_data.loc[selected_indices]
                y_target_select = y_target_select.loc[selected_indices]
            else:
                X_data_selection = X_data.copy()

            # Split the dataset into training and test sets
            logging.info('Splitting the dataset into training and test sets...')
            X_train_all_features, X_test_all_features, y_train_select, y_test_select = train_test_split(X_data_selection, y_target_select, train_size=train_split, random_state=42)
            logging.info(f'Size of X_train {X_train_all_features.shape} and size of y_train {y_train_select.shape}')
            logging.info(f'Size of X_test {X_test_all_features.shape} and size of y_test {y_test_select.shape}')

            # Standardise the data
            logging.info('Standardising the descriptor data...')
            X_train, X_test = data_standardisation(X_train_all_features, X_test_all_features)

            # Perform feature selection to select the k best features
            if len(X_train_all_features.columns.tolist()) > 1000:
                X_train, X_test = perform_feature_selection(X_train_all_features, y_train_select, X_test_all_features, k=k_features, data_modalities=list(x_prefixes & col_prefixes), save_path=target_directory)
                logging.info(f'New size of X_train {X_train.shape} and X_test {X_test.shape}')
            else:
                X_train = X_train_all_features.copy()
                X_test = X_test_all_features.copy()

            for algorithm in algorithms_list:

                # Create the algorithm directory
                algorithm_directory = target_directory+'/'+algorithm
                if not os.path.exists(algorithm_directory):
                    os.makedirs(algorithm_directory)
                
                logging.info(f'\n=================={algorithm}===================')
                start_time = time.time()

                # Check if the combination target-algorithm already exists in the results file
                if (pharmaco_target[3:], algorithm) in existing_combinations:
                    continue

                # Create an Optuna study to perform the hyperparameter optimization for the corresponding algorithm
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda trial: objective(trial, algorithm, X_train, y_train_select), n_trials=num_trials) # n_jobs=18

                # Save the study object to a file
                with open(algorithm_directory+'/optuna_study.pickle', 'wb') as f:
                    pickle.dump(study, f)

                # Save the Optuna results 
                trials_df = study.trials_dataframe()
                trials_df.to_csv(algorithm_directory+'/optuna_trials.tsv', sep='\t', index=False)

                # Retrieve both training and validation scores for each trial
                train_scores = [trial.user_attrs['train_score'] if 'train_score' in trial.user_attrs else 0 for trial in study.trials]
                val_scores = [trial.value for trial in study.trials]
                val_scores = [0 if train_score == 0 else val_score for train_score, val_score in zip(train_scores, val_scores)] # set val_score to 0 if the trail has been pruned

                if verbose_optuna == True:
                    # Plot the optimization history
                    optuna.visualization.plot_optimization_history(study).show()
                    # Plot the parallel coordinate plot
                    optuna.visualization.plot_parallel_coordinate(study).show()
                    # Plot the pruning history
                    optuna.visualization.plot_intermediate_values(study).show()

                # Plot the training vs. validation scores curve
                plot_train_vs_validation_loss(train_scores, val_scores, study.best_trial.number, show_plot=plot_loss_model, save_path=algorithm_directory)

                logging.info(f'Best trial: {str(study.best_trial.number)}')
                logging.info(f'Best params: {str(study.best_params)}')

                # Re-define the model with the best hyperparameter combination
                if algorithm == 'GradientBoosting':
                    model = GradientBoostingClassifier(**study.best_params, criterion='friedman_mse', loss='log_loss', verbose=0, random_state=42)
                elif algorithm == 'XGBoost':
                    model = XGBClassifier(**study.best_params, objective='binary:logistic', booster='gbtree', tree_method='exact', verbosity=0, random_state=42)

                # Train the model
                model.fit(X_train, y_train_select, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_select))

                # Save the trained model to a file
                with open(algorithm_directory+'/model.pickle', "wb") as fout:
                    pickle.dump(model, fout)
                
                # Get predictions on the test set
                y_test_pred_proba = model.predict_proba(X_test)
                # Convert them to a DataFrame 
                prob_df = pd.DataFrame(y_test_pred_proba)
                # Extract the compound identifiers and convert them to a DataFrame
                inchis_df = pd.DataFrame({'CPD_INCHIKEY': data.loc[y_test_select.index, 'CPD_INCHIKEY']})
                # Concatenate both DataFrames
                inchis_prob_df = pd.concat([inchis_df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
                # Save it to a CSV file
                inchis_prob_df.to_csv(algorithm_directory+'/test_predicted_probabilities.csv', index=False)

                # Set the threshold and classify the predicted proabilities
                threshold = 0.5
                y_test_pred_class = (y_test_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the test set
                tn_test, fp_test, fn_test, tp_test, balanced_accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, mcc_test, avg_precision_test = evaluate_model_performance(y_test_select, y_test_pred_class)

                # Get predictions on the training set
                y_train_pred_proba = model.predict_proba(X_train)
                # Classify the predicted proabilities
                y_train_pred_class = (y_train_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the training set
                tn_train, fp_train, fn_train, tp_train, balanced_accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, mcc_train, avg_precision_train = evaluate_model_performance(y_train_select, y_train_pred_class)

                end_time = time.time() 
                total_time = (end_time - start_time) / 60

                logging.info(f'Time: {total_time:.2f} minutes')
                logging.info(f'Balanced Accuracy (train) = {balanced_accuracy_train} --- Precision (train) = {precision_train} --- Recall (train)= {recall_train} --- F1-score (train)= {f1_train} --- ROC AUC score (train) = {roc_auc_train} --- MCC (train)= {mcc_train} --- PR AUC score (train)= {avg_precision_train}')
                logging.info(f'Balanced Accuracy (test) = {balanced_accuracy_test} --- Precision (test) = {precision_test} --- Recall (test)= {recall_test} --- F1-score (test)= {f1_test} --- ROC AUC score (test) = {roc_auc_test} --- MCC (test)= {mcc_test} --- PR AUC score (test)= {avg_precision_test}')
                logging.info(f'F1-score (validation): {study.best_trial.value}')

                # Construct the 'data' column 
                data_column = ', '.join(map(str, [data_dict[prefix] for prefix in x_prefixes & col_prefixes]))

                # Write values to the results file
                with open('results/'+results_filename, 'a') as resultsfile:
                    resultsfile.write(f'{data_column}\t{pharmaco_target[3:]}\t{positive_counts} ({round((positive_counts/(positive_counts+negative_counts))*100, 2)}%)\t{negative_counts} ({round((negative_counts/positive_counts+negative_counts)*100, 2)}%)\t{round(positive_counts/negative_counts, 4)}\t'
                                      f'{algorithm}\t{num_trials}\t{study.best_trial.number}\t{tp_train}\t{fp_train}\t{fn_train}\t{tn_train}\t{tp_test}\t{fp_test}\t{fn_test}\t{tn_test}\t'
                                      f'{balanced_accuracy_train}\t{balanced_accuracy_test}\t{precision_train}\t{precision_test}\t{recall_train}\t{recall_test}\t{f1_train}\t{study.best_trial.value}\t{f1_test}\t'
                                      f'{roc_auc_train}\t{roc_auc_test}\t{mcc_train}\t{mcc_test}\t{avg_precision_train}\t{avg_precision_test}\t{total_time}\n')

                # Plot several model evaluation plots
                plot_model_evaluation(pharmaco_target[3:], algorithm, y_train_select, y_train_pred_proba, y_train_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=True)
                plot_model_evaluation(pharmaco_target[3:], algorithm, y_test_select, y_test_pred_proba, y_test_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=False)
            
                if hasattr(model, 'feature_importances_'):
                    # Plot the model feature importances
                    plot_model_feature_importance(pharmaco_target[3:], algorithm, model, X_train, show_plot=plot_feature_importance, save_path=algorithm_directory)

    # Load the results TSV file
    final_df = pd.read_csv('results/'+results_filename, sep='\t')

    # Return the metrics dataframe
    return final_df 

# ============================= LATE DATA FUSION ==============================

def tg_late_data_fusion(data_modalities, target, fusion_method, save_results=False, results_filename=None):
    
    data_modalities_dict = {'cp':'Cell Painting', 'desc':'RDKit 1D descriptors', 'ecfp4':'ECFP4 fingerprints', 'mordred':'Mordred descriptors', 'pc': 'Physicochemical properties'}
    fusion_methods_list = ['average', 'weighted_average', 'voting', 'weighted_voting', 'maximal', 'weighted_maximal']

    # Define the function arguments and their expected data types and allowed values
    args_type = {'data_modalities': list, 'target': Union[str, list], 'fusion_method': str, 'save_results': bool, 'results_filename': Optional[str]}
    
    # Validate argument data types
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__}.")
    
    # Check allowed values for the 'data_modalities' and 'fusion_method' arguments
    if isinstance(target, str):
        target = [target] 
    for modality in data_modalities:
        if modality not in data_modalities_dict.keys():
            raise ValueError(f"'{modality}' is not a valid data modality. It must be one of {data_modalities_dict.keys()} values.")
    if fusion_method not in fusion_methods_list:
        raise ValueError(f"'{fusion_method}' is not a valid fusion method. It must be one of {fusion_methods_list} values.")
        
    # Load the pharmacology data
    logging.info('Loading the pharmacology data...')
    pharmacology_data = pd.read_csv('data/4_data/target_binary_matrix.csv')
    pharmacology_data_columns = [col for col in pharmacology_data.columns.tolist() if col.startswith('tg')]

    # Load the TSV results file for each data modality
    results_df = []
    for modality in data_modalities:
        modality_df = pd.read_csv('data/4_data/'+modality+'_pharmacology_results.tsv', sep='\t')
        results_df. append(modality_df)

    # Create the list of targets to be modeled
    if len(target) == 1 and target[0] == 'all':
        y_features = [tg for tg in pharmacology_data_columns]
    else:
        y_features = ['tg_'+tg for tg in target]

    # Create an empty list to store metrics dictionaries
    metrics_list = []
    # Construct the data column
    data_column = ', '.join(map(str, [data_modalities_dict[modality] for modality in data_modalities]))

    # Iterate over all pharmacology endpoints
    for target in pharmacology_data_columns:

        # Check if the target has to be modeled
        if target in y_features:

            logging.info(f'\n\n·················{target[3:]} ({pharmacology_data_columns.index(target)+1}/{len(pharmacology_data_columns)})·················\n')

            # Initialize two lists to store predicted probabilities and model performances for each modality
            y_test_probs = []
            model_performances = []

            # Iterate over the input data modalities
            for i in range(len(data_modalities)):
                # Select the modality and its results dataframe
                modality = data_modalities[i]
                modality_df = results_df[i]
                # Select the rows corresponding to the pharmacology endpoint results
                target_results = modality_df.loc[modality_df['target'] == target[3:]]
                # Get the model with the maximum MCC validation score
                best_model = target_results.loc[target_results['f1_score(validation)'].idxmax(), 'ML_model']
                # Specify the path to the CSV file
                model_file = 'log/'+modality+'_logs/'+target[3:]+'/'+best_model+'/test_predicted_probabilities.csv'
                # Load predicted probabilities from the CSV file
                y_test_modality_probs = pd.read_csv(model_file, header=0)
                # Append the modality probabilities and the best model performance to the lists
                y_test_probs.append(y_test_modality_probs.iloc[:,1:])
                model_performances.append(target_results['f1_score(validation)'].max())

            # Get the InChiKey of the test compounds
            y_test_inchis = y_test_modality_probs.iloc[:,0]
            # Select the target data corresponding to the test compounds
            test_pharmacology_data = pharmacology_data.set_index('CPD_INCHIKEY').loc[y_test_inchis].reset_index()
            y_test_select = test_pharmacology_data.loc[:, target]

            if all(len(y_test_modality_probs) == len(y_test_probs[0]) for y_test_modality_probs in y_test_probs):
                # Perform the selected fusion method
                logging.info('Performing late fusion employing the selected method...')

                if fusion_method == 'average':
                    # Stack the probabilities across modalities along the third axis
                    y_test_probs = np.stack(y_test_probs, axis=2)
                    # Compute the mean probability across data modalities for each sample
                    y_test_final_probs = np.mean(y_test_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                elif fusion_method == 'weighted_average':
                    # Initialize a list to store weighted probabilities
                    y_weighted_probs = []
                    # Iterate over the probabilities and model performance of each probability
                    for probs, weight in zip(y_test_probs, model_performances):
                        # Perform element-wise multiplication
                        weighted_probs = probs * weight
                        # Append the weighted probabilities to the list
                        y_weighted_probs.append(weighted_probs)
                    # Stack the probabilities across modalities along the third axis
                    y_weighted_probs = np.stack(y_weighted_probs, axis=2)
                    # Compute the weighted mean probability across data modalities for each sample
                    y_test_final_probs = np.mean(y_weighted_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                elif fusion_method == 'voting':
                    # Initialize a list to store predicted classes
                    y_test_classes = []
                    for probs in y_test_probs:
                        # Classify the predicted probabilities
                        pred_classes = (np.array(probs)[:, 1] >= 0.5).astype(int)
                        # Append the predicted classes to the list
                        y_test_classes.append(pred_classes)
                    # Stack the classes across modalities along the third axis
                    y_test_classes = np.stack(y_test_classes, axis=1)
                    # Compute the mean class probability across data modalities for each sample
                    y_test_positive_class_probs = np.mean(y_test_classes, axis=1)

                elif fusion_method == 'weighted_voting':
                    # Initialize a list to store weighted predicted classes
                    y_weighted_classes = []
                    for probs, weight in zip(y_test_probs, model_performances):
                        # Classify the predicted probabilities
                        pred_classes = (np.array(probs)[:, 1] >= 0.5).astype(int)
                        # Perform element-wise multiplication
                        weighted_classes = pred_classes * weight
                        # Append the weighted classes to the list
                        y_weighted_classes.append(weighted_classes)
                    # Stack the probabilities across modalities along the third axis
                    y_weighted_classes = np.stack(y_weighted_classes, axis=1)
                    # Compute the weighted mean class probability across data modalities for each sample
                    y_test_positive_class_probs = np.mean(y_weighted_classes, axis=1)

                elif fusion_method == 'maximal':
                    # Stack the probabilities across modalities along the third axis
                    y_test_probs = np.stack(y_test_probs, axis=2)
                    # Compute the maximal probability across data modalities for each sample
                    y_test_final_probs = np.max(y_test_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                elif fusion_method == 'weighted_maximal':
                    # Initialize a list to store weighted probabilities
                    y_weighted_probs = []
                    # Iterate over the probabilities and model performance of each probability
                    for probs, weight in zip(y_test_probs, model_performances):
                        # Perform element-wise multiplication
                        weighted_probs = probs * weight
                        # Append the weighted probabilities to the list
                        y_weighted_probs.append(weighted_probs)
                    # Stack the probabilities across modalities along the third axis
                    y_weighted_probs = np.stack(y_weighted_probs, axis=2)
                    # Compute the weighted maximmal probability across data modalities for each sample
                    y_test_final_probs = np.max(y_weighted_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                # Calculate the average precision from prediction scores
                avg_precision = metrics.average_precision_score(y_test_select, y_test_positive_class_probs)
                roc_auc = metrics.roc_auc_score(y_test_select, y_test_positive_class_probs)

                # Add the computed evaluation metrics to the dataframe
                metrics_list.append({'data': data_column, 'target': target[3:], 'roc_auc_score': roc_auc, 'pr_auc_score': avg_precision})
                
            else: 
                logging.debug(f'Error: The test set\'s dimensionality (i.e., number of compounds) varies across different data modalities.')
                # Add the computed evaluation metrics to the dataframe
                metrics_list.append({'data': data_column, 'target': target[3:], 'roc_auc_score': np.nan, 'pr_auc_score': np.nan})

    # Create the DataFrame containing the metrics data
    metrics_df = pd.DataFrame(metrics_list)

    # Save the results in a TSV file
    if save_results == True: 
            if results_filename.endswith('tsv'):
                metrics_df.to_csv('results/'+results_filename, sep='\t', index=False)
            elif results_filename.endswith('csv'):
                metrics_df.to_csv('results/'+results_filename, index=False)

    # Return the metrics dataframe
    return metrics_df 

###############################################################################
# ========================= Safety Predictive Models ==========================
###############################################################################

def plot_endpoint_distribution(zero_counts, one_counts, endpoint_id, show_plot=False, save_path=None):
    # Create a pie chart
    labels = ['0', '1']
    sizes = [zero_counts, one_counts]
    colors = ['lightcoral', 'lightskyblue']

    plt.figure(figsize=(10,6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  
    plt.title(f'{endpoint_id} safety endpoint --- Class Distribution')

    if save_path:
        plt.savefig(save_path+'/endpoint_distribution_plot.png', format='png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def se_model_training_and_evaluation(data, endpoint, check_endpoint_distribution=False, optuna_trials=50, verbose_optuna=False, plot_loss_model=False, plot_results=False, plot_feature_importance=False, results_filename=None):
    
    data_dict = {'cp':'Cell Painting', 'desc':'RDKit 1D descriptors', 'ecfp4':'ECFP4 fingerprints', 'mordred':'Mordred descriptors', 'pc': 'Physicochemical properties'}

    # Define the function arguments and their expected data types and allowed values
    args_type = {'check_endpoint_distribution': bool, 'optuna_trials': int, 'verbose_optuna': bool, 'plot_loss_model': bool, 'plot_results': bool, 'plot_feature_importance': bool, 'results_filename': Optional[str]}
    
    endpoint_values = [col_name[3:] for col_name in data.columns.tolist() if col_name.startswith('se_')] + ['all']

    # Validate argument data types
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value.")
    
    # Validate 'endpoint' argument
    if isinstance(endpoint, str):
        endpoint = [endpoint] 
    elif not isinstance(endpoint, list):
        raise TypeError("'endpoint' must be either a string or a list of strings.")
    
    # Check allowed values for the 'endpoint' argument
    for ep in endpoint:
        if ep not in endpoint_values:
            raise ValueError(f"'{ep}' is not a valid endpoint. It must be one of {endpoint_values} values.")
    
    # Define the prefix set for X and the prefix for Y
    x_prefixes = {'cp', 'ecfp4', 'desc', 'mordred', 'pc'} 
    y_prefix = 'se'

    # Determine the X prefix 
    col_prefixes = set([col_name.split('_')[0] for col_name in data.columns.tolist()])
    x_prefix = '|'.join(x_prefixes & col_prefixes)

    # Split the dataset into training and test sets
    logging.info('Splitting the dataset into training and test sets...')
    X_train, X_test, y_train, y_test = train_test_split(data.filter(regex=f'^{x_prefix}', axis=1), data.filter(regex=f'^{y_prefix}', axis=1), train_size=0.8, random_state=42)
    logging.info(f'Size of X_train {X_train.shape} and size of y_train {y_train.shape}')
    logging.info(f'Size of X_test {X_test.shape} and size of y_test {y_test.shape}')

    # Standardise the data
    logging.info('Standardising the descriptor data...')
    X_train, X_test = data_standardisation(X_train, X_test)

    # Define the list of algorithms and set the number of trials
    algorithms_list = ['XGBoost'] # 'GradientBoosting'
    num_trials = optuna_trials

    # Create the list of safety endpoints to be modeled
    if len(endpoint) == 1:
        if endpoint[0] == 'all':
            y_features = [ep for ep in y_train.columns.tolist()]
        else:
            y_features = ['se_'+endpoint[0]]
    else:
        y_features = ['se_'+ep for ep in endpoint]

    # Create the descriptor directory
    input_data_directory = 'log/'+'_'.join(list(x_prefixes & col_prefixes))+'_logs'
    if not os.path.exists(input_data_directory):
        logging.info('Creating a new directory to save Optuna results and additional plots...')
        os.makedirs(input_data_directory)

    # Write the column names in the results file
    if not os.path.exists('results/'+results_filename):
        with open('results/'+results_filename, 'a') as resultsfile:
            resultsfile.write('data\tsafety_endpoint\tpositive_counts\tnegative_counts\tML_model\tnum_trials\tbest_trial\t'
                            'tp(train)\tfp(train)\tfn(train)\ttn(train)\ttp\tfp\tfn\ttn\tbalanced_accuracy(train)\tbalanced_accuracy\tprecision(train)\tprecision\trecall(train)\trecall\t'
                            'f1_score(train)\tf1_score(validation)\tf1_score\troc_auc_score(train)\troc_auc_score\tmcc(train)\tmcc\tpr_auc_score(train)\tpr_auc_score\ttime\n')
    
    # Initialize a set to store the combinations of endpoint-algorithm already present in the results file
    existing_combinations = set()
    # Read the TSV file and parse combinations
    with open('results/'+results_filename, 'r') as resultsfile:
        for line in resultsfile:
            columns = line.split('\t')
            endpoint = columns[1]
            algorithm = columns[4]
            existing_combinations.add((endpoint, algorithm))

    # Iterate over all safety endpoints
    for safety_endpoint in y_train.columns.tolist():

        # Check if the endpoint has to be modeled
        if safety_endpoint in y_features:

            # Create the endpoint directory
            endpoint_directory = input_data_directory+'/'+safety_endpoint[3:]
            if not os.path.exists(endpoint_directory):
                os.makedirs(endpoint_directory)

            logging.info(f'\n\n·················{safety_endpoint[3:]} ({y_train.columns.tolist().index(safety_endpoint)+1}/{len(y_train.columns)})·················\n')

            # Select the safety endpoint in both training and test sets
            y_train_select = y_train.loc[:, safety_endpoint]
            y_test_select = y_test.loc[:, safety_endpoint]

            # Count the instances of each output class
            y_values = y_train_select.tolist() + y_test_select.tolist()
            positive_counts = y_values.count(1)
            negative_counts = y_values.count(0)

            # Plot the endpoint values distribution
            plot_endpoint_distribution(negative_counts, positive_counts, safety_endpoint[3:], show_plot=check_endpoint_distribution, save_path=endpoint_directory)

            for algorithm in algorithms_list:

                # Create the endpoint directory
                algorithm_directory = endpoint_directory+'/'+algorithm
                if not os.path.exists(algorithm_directory):
                    os.makedirs(algorithm_directory)

                logging.info(f'\n=================={algorithm}===================')
                start_time = time.time()

                # Check if the combination endpoint-algorithm already exists in the results file
                if (safety_endpoint[3:], algorithm) in existing_combinations:
                    continue

                # Create an Optuna study to perform the hyperparameter optimization for the corresponding algorithm
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda trial: objective(trial, algorithm, X_train, y_train_select), n_trials=num_trials) 

                # Save the study object to a file
                with open(algorithm_directory+'/optuna_study.pickle', 'wb') as f:
                    pickle.dump(study, f)

                # Save the Optuna results 
                trials_df = study.trials_dataframe()
                trials_df.to_csv(algorithm_directory+'/optuna_trials.tsv', sep='\t', index=False)

                # Retrieve both training and validation scores for each trial
                train_scores = [trial.user_attrs['train_score'] if 'train_score' in trial.user_attrs else 0 for trial in study.trials]
                val_scores = [trial.value for trial in study.trials]
                val_scores = [0 if train_score == 0 else val_score for train_score, val_score in zip(train_scores, val_scores)] # set val_score to 0 if the trail has been pruned

                if verbose_optuna == True:
                    # Plot the optimization history
                    optuna.visualization.plot_optimization_history(study).show()
                    # Plot the parallel coordinate plot
                    optuna.visualization.plot_parallel_coordinate(study).show()
                    # Plot the pruning history
                    optuna.visualization.plot_intermediate_values(study).show()

                # Plot the training vs. validation scores curve
                plot_train_vs_validation_loss(train_scores, val_scores, study.best_trial.number, show_plot=plot_loss_model, save_path=algorithm_directory)
                
                logging.info(f'Best trial: {str(study.best_trial.number)}')
                logging.info(f'Best params: {str(study.best_params)}')

                # Re-define the model with the best hyperparameter combination
                if algorithm == 'GradientBoosting':
                    model = GradientBoostingClassifier(**study.best_params, criterion='friedman_mse', loss='log_loss', verbose=0, random_state=42)
                elif algorithm == 'XGBoost':
                    model = XGBClassifier(**study.best_params, objective='binary:logistic', booster='gbtree', tree_method='exact', verbosity=0, random_state=42)

                # Train the model
                model.fit(X_train, y_train_select, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_select))

                # Save the trained model to a file
                with open(algorithm_directory+'/model.pickle', "wb") as fout:
                    pickle.dump(model, fout)
                
                # Get predictions on the test set
                y_test_pred_proba = model.predict_proba(X_test)
                # Convert them to a DataFrame 
                prob_df = pd.DataFrame(y_test_pred_proba)
                # Extract the compound identifiers and convert them to a DataFrame
                inchis_df = pd.DataFrame({'CPD_INCHIKEY': data.loc[y_test_select.index, 'CPD_INCHIKEY']})
                # Concatenate both DataFrames
                inchis_prob_df = pd.concat([inchis_df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
                # Save it to a CSV file
                inchis_prob_df.to_csv(algorithm_directory+'/test_predicted_probabilities.csv', index=False)

                # Set the threshold and classify the predicted proabilities
                threshold = 0.5
                y_test_pred_class = (y_test_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the test set
                tn_test, fp_test, fn_test, tp_test, balanced_accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, mcc_test, avg_precision_test = evaluate_model_performance(y_test_select, y_test_pred_class)

                # Get predictions on the training set
                y_train_pred_proba = model.predict_proba(X_train)
                # Classify the predicted proabilities
                y_train_pred_class = (y_train_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the training set
                tn_train, fp_train, fn_train, tp_train, balanced_accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, mcc_train, avg_precision_train = evaluate_model_performance(y_train_select, y_train_pred_class)

                end_time = time.time() 
                total_time = (end_time - start_time) / 60

                logging.info(f'Time: {total_time:.2f} minutes')
                logging.info(f'Balanced Accuracy (train) = {balanced_accuracy_train} --- Precision (train) = {precision_train} --- Recall (train)= {recall_train} --- F1 score (train)= {f1_train} --- ROC AUC score (train) = {roc_auc_train} --- MCC (train)= {mcc_train} --- PR AUC score (train)= {avg_precision_train}')
                logging.info(f'Balanced Accuracy (test) = {balanced_accuracy_test} --- Precision (test) = {precision_test} --- Recall (test)= {recall_test} --- F1 score (test)= {f1_test} --- ROC AUC score (test) = {roc_auc_test} --- MCC (test)= {mcc_test} --- PR AUC score (test)= {avg_precision_test}')
                logging.info(f'F1-score (validation): {study.best_trial.value}')

                # Construct the data column
                data_column = ', '.join(map(str, [data_dict[prefix] for prefix in x_prefixes & col_prefixes]))

                # Write values to the results file
                with open('results/'+results_filename, 'a') as resultsfile:
                    resultsfile.write(f'{data_column}\t{safety_endpoint[3:]}\t{positive_counts}({round((positive_counts/len(y_values))*100, 2)}%)\t{negative_counts} ({round((negative_counts/len(y_values))*100, 2)}%)\t'
                                      f'{algorithm}\t{num_trials}\t{study.best_trial.number}\t{tp_train}\t{fp_train}\t{fn_train}\t{tn_train}\t{tp_test}\t{fp_test}\t{fn_test}\t{tn_test}\t'
                                      f'{balanced_accuracy_train}\t{balanced_accuracy_test}\t{precision_train}\t{precision_test}\t{recall_train}\t{recall_test}\t{f1_train}\t{study.best_trial.value}\t{f1_test}\t'
                                      f'{roc_auc_train}\t{roc_auc_test}\t{mcc_train}\t{mcc_test}\t{avg_precision_train}\t{avg_precision_test}\t{total_time}\n')

                # Plot several model evaluation plots
                plot_model_evaluation(safety_endpoint[3:], algorithm, y_train_select, y_train_pred_proba, y_train_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=True)
                plot_model_evaluation(safety_endpoint[3:], algorithm, y_test_select, y_test_pred_proba, y_test_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=False)
                
                if hasattr(model, 'feature_importances_'):
                    # Plot the model feature importances
                    plot_model_feature_importance(safety_endpoint[3:], algorithm, model, X_train, show_plot=plot_feature_importance, save_path=algorithm_directory)

    # Load the results TSV file
    final_df = pd.read_csv('results/'+results_filename, sep='\t')

    return final_df

# ======================= Analysis of Individual Models =======================

def read_meddra_tree(meddra_tree_path):
    med_df = pd.read_csv(meddra_tree_path, sep='\t' if '.tsv' in meddra_tree_path else ',')
    med_df = med_df[med_df['tree_id'] == 'MedDRA'].reset_index(drop=True)
    return med_df

def get_pt2hlt(meddra_tree_path, reverse=False):

    med_df = read_meddra_tree(meddra_tree_path)

    pt2hlt = {}
    for path in med_df['path']:
        path = path.split('.')
        if len(path)== 4:
            pt = path[-1]
            hlt = path[-2]
            if reverse:
                if hlt not in pt2hlt:
                    pt2hlt[hlt] = set([])
                pt2hlt[hlt].add(pt)
            else:
                if pt not in pt2hlt:
                    pt2hlt[pt] = set([])
                pt2hlt[pt].add(hlt)

    return pt2hlt

def get_pt2hlgt(meddra_tree_path, reverse=False):

    med_df = read_meddra_tree(meddra_tree_path)

    pt2hlgt = {}
    for path in med_df['path']:
        path = path.split('.')
        if len(path)== 4:
            pt = path[-1]
            hlgt = path[-3]
            if reverse:
                if hlgt not in pt2hlgt:
                    pt2hlgt[hlgt] = set([])
                pt2hlgt[hlgt].add(pt)
            else:
                if pt not in pt2hlgt:
                    pt2hlgt[pt] = set([])
                pt2hlgt[pt].add(hlgt)

    return pt2hlgt

def get_pt2soc(meddra_tree_path, reverse=False):

    med_df = read_meddra_tree(meddra_tree_path)

    pt2soc = {}
    for path in med_df['path']:
        path = path.split('.')
        if len(path)== 4:
            pt = path[-1]
            soc = path[-4].lstrip('M_')
            if reverse:
                if soc not in pt2soc:
                    pt2soc[soc] = set([])
                pt2soc[soc].add(pt)
            else:
                if pt not in pt2soc:
                    pt2soc[pt] = set([])
                pt2soc[pt].add(soc)

    return pt2soc

def classify_pt_to_meddra(safety_df):
    # Load the PT annotations
    pt_data = pd.read_csv('data/5_data/PT_id_label.csv')

    # Get the label of the top PTs
    safety_df = pd.merge(safety_df[['safety_endpoint', 'mcc']], pt_data, left_on='safety_endpoint', right_on='term_id')
    safety_df = safety_df.drop('term_id', axis=1) # remove the 'term_id' column
    safety_df.rename(columns={'safety_endpoint':'pt_id'}, inplace=True) # rename the 'safety_endpoint' column
    safety_df = safety_df.drop_duplicates() # drop duplicated rows

    # Map the PTs to HLT, HLGT and SOC
    pt_to_hlt = get_pt2hlt('data/5_data/MedDRA_db.csv')
    pt_to_hlgt = get_pt2hlgt('data/5_data/MedDRA_db.csv')
    pt_to_soc = get_pt2soc('data/5_data/MedDRA_db.csv')

    # Load the HLT, HLGT and SOC annotations
    hlt_data = pd.read_csv('data/5_data/HLT_id_label.csv', dtype=str)
    hlgt_data = pd.read_csv('data/5_data/HLGT_id_label.csv', dtype=str)
    soc_data = pd.read_csv('data/5_data/SOC_id_label.csv', dtype=str)

    # Map the top PTs to their corresponding HLT, HLGT and SOC
    data_list = []
    for index, row in safety_df.iterrows():
        pt_id = str(row['pt_id'])
        pt_name = row['label']
        mcc = row['mcc']

        if pt_id in pt_to_hlt:
            hlt_values = pt_to_hlt[pt_id]
            for hlt_id in hlt_values:

                if pt_id in pt_to_hlgt:
                    hlgt_values = pt_to_hlgt[pt_id]
                    for hlgt_id in hlgt_values:

                        if pt_id in pt_to_soc:
                            soc_values = pt_to_soc[pt_id]
                            for soc_id in soc_values:
                                data_list.append({'pt_id':pt_id, 'pt_name':pt_name, 'hlt_id':hlt_id, 'hlgt_id':hlgt_id, 'soc_id':soc_id, 'mcc':mcc})

                        else:
                            data_list.append({'pt_id':pt_id, 'pt_name':pt_name, 'hlt_id':hlt_id, 'hlgt_id':hlgt_id, 'soc_id':np.nan, 'mcc':mcc})

                else:
                    data_list.append({'pt_id':pt_id, 'pt_name':pt_name, 'hlt_id':hlt_id, 'hlgt_id':np.nan, 'soc_id':np.nan, 'mcc':mcc})
        
        else:
            data_list.append({'pt_id':pt_id, 'pt_name':pt_name, 'hlt_id':np.nan, 'hlgt_id':np.nan, 'soc_id':np.nan, 'mcc':mcc})

    # Create the DataFrame containing the MedDRA data
    pt_to_meddra_df = pd.DataFrame(data_list)

    # Get the label of the HLT, HLGT and SOC of the top PTs
    pt_to_meddra_df = pd.merge(pt_to_meddra_df, hlt_data, left_on='hlt_id', right_on='term_id')
    pt_to_meddra_df = pt_to_meddra_df.drop('term_id', axis=1) # remove the 'term_id' column
    pt_to_meddra_df.rename(columns={'label':'hlt_name'}, inplace=True) # rename the 'label' column
    pt_to_meddra_df = pt_to_meddra_df.drop_duplicates() # drop duplicated rows

    pt_to_meddra_df = pd.merge(pt_to_meddra_df, hlgt_data, left_on='hlgt_id', right_on='term_id')
    pt_to_meddra_df = pt_to_meddra_df.drop('term_id', axis=1) # remove the 'term_id' column
    pt_to_meddra_df.rename(columns={'label':'hlgt_name'}, inplace=True) # rename the 'label' column
    pt_to_meddra_df = pt_to_meddra_df.drop_duplicates() # drop duplicated rows

    pt_to_meddra_df = pd.merge(pt_to_meddra_df, soc_data, left_on='soc_id', right_on='term_id')
    pt_to_meddra_df = pt_to_meddra_df.drop('term_id', axis=1) # remove the 'term_id' column
    pt_to_meddra_df.rename(columns={'label':'soc_name'}, inplace=True) # rename the 'label' column
    pt_to_meddra_df = pt_to_meddra_df.drop_duplicates() # drop duplicated rows

    # Shorten the SOC names
    pattern = r'\s*(,|disorders|procedures|complications)\s*$' # pattern to match words to remove at the end of the string
    pt_to_meddra_df['soc_name'] = pt_to_meddra_df['soc_name'].apply(lambda x: re.sub(pattern, '', x))
    pt_to_meddra_df['soc_name'] = pt_to_meddra_df['soc_name'].apply(lambda x: 'Neoplasms' if x.startswith('Neoplasms') else x)
        # kepp only the 'Neoplasms' word for the corresponding SOC

    # Rearrange the order of the columns
    pt_to_meddra_df = pt_to_meddra_df[['pt_id','pt_name','hlt_id','hlt_name','hlgt_id','hlgt_name','soc_id','soc_name','mcc']]

    return pt_to_meddra_df

def fisher_exact_test_on_MedDRA_terms(pt_to_meddra_subset, pt_to_meddra_all, meddra_level):

    # Get the MedDRA terms to be analysed
    meddra_terms = pt_to_meddra_subset[meddra_level+'_id'].unique().tolist()

    # Create an empty list to store the results
    fisher_results = []

    # Iterate the selected terms
    for term in meddra_terms:
        # Get the number of top PTs belonging to the term
        top_pt_term = set(pt_to_meddra_subset['pt_id'].loc[pt_to_meddra_subset[meddra_level+'_id'] == term].tolist())
        top_pt_term_counts = len(top_pt_term)
        # Get the number of top PTs not belonging to the term
        top_pt_no_term_counts = len(set(pt_to_meddra_subset['pt_id'].tolist())) - top_pt_term_counts
        # Get the number of general PTs belonging to the term
        all_pt_term = set(pt_to_meddra_all['pt_id'].loc[pt_to_meddra_all[meddra_level+'_id'] == term].tolist())
        all_pt_term_counts = len(all_pt_term) - top_pt_term_counts
        # Get the number of general PTs not belonging to the term
        all_pt_no_term_counts = len(set(pt_to_meddra_all['pt_id'].tolist())) - all_pt_term_counts - top_pt_no_term_counts

        # Build the contingency table
        contingency_table = np.array([[top_pt_term_counts, top_pt_no_term_counts], [all_pt_term_counts, all_pt_no_term_counts]])
        # Perform the Fisher exact test
        odd_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')

        # Append the results to the list
        fisher_results.append({f'{meddra_level}_id':term, f'{meddra_level}_name':pt_to_meddra_subset[meddra_level+'_name'].loc[pt_to_meddra_subset[meddra_level+'_id'] == term].values[0], 
                               'contingency_table':contingency_table, 'odd-ratio':odd_ratio, 'p-value':p_value, 'intersections':', '.join([str(item) for item in top_pt_term])})

    # Transfrom the results to a Pandas Dataframe
    results_df = pd.DataFrame(data=fisher_results)

    # Correct for multiple testing using the FDR method
    adjusted_pvalues = smt.multipletests(results_df['p-value'].tolist(), method='fdr_bh')
    results_df['p-value_adjusted'] = adjusted_pvalues[1]

    # Add a column to indicate if the MedDRA term is statistically significant 
    results_df['significance_0.05'] = results_df['p-value_adjusted'].apply(lambda x: '*' if x < 0.05 else '')
    results_df['significance_0.1'] = results_df['p-value_adjusted'].apply(lambda x: '*' if x < 0.1 else '')

    # Return the results
    return results_df

# ============================= EARLY DATA FUSION =============================

def se_early_fusion_model_training_and_evaluation(data, endpoint, k_features=1000, check_endpoint_distribution=False, optuna_trials=50, verbose_optuna=False, plot_loss_model=False, plot_results=False, plot_feature_importance=False, results_filename=None):
    
    data_dict = {'cp':'Cell Painting', 'desc':'RDKit 1D descriptors', 'ecfp4':'ECFP4 fingerprints', 'mordred':'Mordred descriptors', 'pc': 'Physicochemical properties'}

    # Define the function arguments and their expected data types and allowed values
    args_type = {'k_features': int, 'check_endpoint_distribution': bool, 'optuna_trials': int, 'verbose_optuna': bool, 'plot_loss_model': bool, 'plot_results': bool, 'plot_feature_importance': bool, 'results_filename': Optional[str]}
    
    endpoint_values = [col_name[3:] for col_name in data.columns.tolist() if col_name.startswith('se_')] + ['all']

    # Validate argument data types
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__} value.")
    
    # Validate 'endpoint' argument
    if isinstance(endpoint, str):
        endpoint = [endpoint] 
    elif not isinstance(endpoint, list):
        raise TypeError("'endpoint' must be either a string or a list of strings.")
    
    # Check allowed values for the 'endpoint' argument
    for ep in endpoint:
        if ep not in endpoint_values:
            raise ValueError(f"'{ep}' is not a valid endpoint. It must be one of {endpoint_values} values.")
            
    # Define the prefix set for X and the prefix for Y
    x_prefixes = {'cp', 'ecfp4', 'desc', 'mordred', 'pc'} 
    y_prefix = 'se'

    # Determine the X prefix 
    col_prefixes = set([col_name.split('_')[0] for col_name in data.columns.tolist()])
    x_prefix = '|'.join(x_prefixes & col_prefixes)

    # Split the dataset into training and test sets
    logging.info('Splitting the dataset into training and test sets...')
    X_train_all_features, X_test_all_features, y_train, y_test = train_test_split(data.filter(regex=f'^{x_prefix}', axis=1), data.filter(regex=f'^{y_prefix}', axis=1), train_size=0.8, random_state=42)
    logging.info(f'Size of X_train {X_train_all_features.shape} and size of y_train {y_train.shape}')
    logging.info(f'Size of X_test {X_test_all_features.shape} and size of y_test {y_test.shape}')

    # Standardise the data
    logging.info('Standardising the descriptor data...')
    X_train_all_features, X_test_all_features= data_standardisation(X_train_all_features, X_test_all_features)

    # Define the list of algorithms and set the number of trials
    algorithms_list = ['XGBoost'] # 'GradientBoositng'
    num_trials = optuna_trials

    # Create the list of safety endpoints to be modeled
    if len(endpoint) == 1:
        if endpoint[0] == 'all':
            y_features = [ep for ep in y_train.columns.tolist()]
        else:
            y_features = ['se_'+endpoint[0]]
    else:
        y_features = ['se_'+ep for ep in endpoint]

    # Create the descriptor directory
    input_data_directory = 'log/'+'_'.join(list(x_prefixes & col_prefixes))+'_logs'
    if not os.path.exists(input_data_directory):
        logging.info('Creating a new directory to save Optuna results and additional plots...')
        os.makedirs(input_data_directory)

    # Write the column names in the results file
    if not os.path.exists('results/'+results_filename):
        with open('results/'+results_filename, 'a') as resultsfile:
            resultsfile.write('data\tsafety_endpoint\tpositive_counts\tnegative_counts\tML_model\tnum_trials\tbest_trial\t'
                            'tp(train)\tfp(train)\tfn(train)\ttn(train)\ttp\tfp\tfn\ttn\tbalanced_accuracy(train)\tbalanced_accuracy\tprecision(train)\tprecision\trecall(train)\trecall\t'
                            'f1_score(train)\tf1_score(validation)\tf1_score\troc_auc_score(train)\troc_auc_score\tmcc(train)\tmcc\tpr_auc_score(train)\tpr_auc_score\ttime\n')
    
    # Initialize a set to store the combinations of endpoint-algorithm already present in the results file
    existing_combinations = set()
    # Read the TSV file and parse combinations
    with open('results/'+results_filename, 'r') as resultsfile:
        for line in resultsfile:
            columns = line.split('\t')
            endpoint = columns[1]
            algorithm = columns[4]
            existing_combinations.add((endpoint, algorithm))

    # Iterate over all safety endpoints
    for safety_endpoint in y_train.columns.tolist():

        # Check if the endpoint has to be modeled
        if safety_endpoint in y_features:

            # Create the endpoint directory
            endpoint_directory = input_data_directory+'/'+safety_endpoint[3:]
            if not os.path.exists(endpoint_directory):
                os.makedirs(endpoint_directory)

            logging.info(f'\n\n·················{safety_endpoint[3:]} ({y_train.columns.tolist().index(safety_endpoint)+1}/{len(y_train.columns)})·················\n')

            # Select the safety endpoint in both training and test sets
            y_train_select = y_train.loc[:, safety_endpoint]
            y_test_select = y_test.loc[:, safety_endpoint]

            # Perform feature selection to select the k best features
            if len(X_train_all_features.columns.tolist()) > 1000:
                X_train, X_test = perform_feature_selection(X_train_all_features, y_train_select, X_test_all_features, k=k_features, data_modalities=list(x_prefixes & col_prefixes), save_path=endpoint_directory)
                logging.info(f'New size of X_train {X_train.shape} and X_test {X_test.shape}')
            else:
                X_train = X_train_all_features.copy()
                X_test = X_test_all_features.copy()
                
            # Count the instances of each output class
            y_values = y_train_select.tolist() + y_test_select.tolist()
            positive_counts = y_values.count(1)
            negative_counts = y_values.count(0)

            # Plot the endpoint values distribution
            plot_endpoint_distribution(negative_counts, positive_counts, safety_endpoint[3:], show_plot=check_endpoint_distribution, save_path=endpoint_directory)

            for algorithm in algorithms_list:

                # Create the endpoint directory
                algorithm_directory = endpoint_directory+'/'+algorithm
                if not os.path.exists(algorithm_directory):
                    os.makedirs(algorithm_directory)

                logging.info(f'\n=================={algorithm}===================')
                start_time = time.time()

                # Check if the combination endpoint-algorithm already exists in the results file
                if (safety_endpoint[3:], algorithm) in existing_combinations:
                    continue

                # Create an Optuna study to perform the hyperparameter optimization for the corresponding algorithm
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda trial: objective(trial, algorithm, X_train, y_train_select), n_trials=num_trials) 

                # Save the study object to a file
                with open(algorithm_directory+'/optuna_study.pickle', 'wb') as f:
                    pickle.dump(study, f)

                # Save the Optuna results 
                trials_df = study.trials_dataframe()
                trials_df.to_csv(algorithm_directory+'/optuna_trials.tsv', sep='\t', index=False)

                # Retrieve both training and validation scores for each trial
                train_scores = [trial.user_attrs['train_score'] if 'train_score' in trial.user_attrs else 0 for trial in study.trials]
                val_scores = [trial.value for trial in study.trials]
                val_scores = [0 if train_score == 0 else val_score for train_score, val_score in zip(train_scores, val_scores)] # set val_score to 0 if the trail has been pruned

                if verbose_optuna == True:
                    # Plot the optimization history
                    optuna.visualization.plot_optimization_history(study).show()
                    # Plot the parallel coordinate plot
                    optuna.visualization.plot_parallel_coordinate(study).show()
                    # Plot the pruning history
                    optuna.visualization.plot_intermediate_values(study).show()

                # Plot the training vs. validation scores curve
                plot_train_vs_validation_loss(train_scores, val_scores, study.best_trial.number, show_plot=plot_loss_model, save_path=algorithm_directory)
                
                logging.info(f'Best trial: {str(study.best_trial.number)}')
                logging.info(f'Best params: {str(study.best_params)}')

                # Re-define the model with the best hyperparameter combination
                if algorithm == 'GradientBoosting':
                    model = GradientBoostingClassifier(**study.best_params, criterion='friedman_mse', loss='log_loss', verbose=0, random_state=42)
                elif algorithm == 'XGBoost':
                    model = XGBClassifier(**study.best_params, objective='binary:logistic', booster='gbtree', tree_method='exact', verbosity=0, random_state=42)

                # Train the model
                model.fit(X_train, y_train_select, sample_weight=compute_sample_weight(class_weight='balanced', y=y_train_select))

                # Save the trained model to a file
                with open(algorithm_directory+'/model.pickle', "wb") as fout:
                    pickle.dump(model, fout)
                
                # Get predictions on the test set
                y_test_pred_proba = model.predict_proba(X_test)
                # Convert them to a DataFrame 
                prob_df = pd.DataFrame(y_test_pred_proba)
                # Extract the compound identifiers and convert them to a DataFrame
                inchis_df = pd.DataFrame({'CPD_INCHIKEY': data.loc[y_test_select.index, 'CPD_INCHIKEY']})
                # Concatenate both DataFrames
                inchis_prob_df = pd.concat([inchis_df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
                # Save it to a CSV file
                inchis_prob_df.to_csv(algorithm_directory+'/test_predicted_probabilities.csv', index=False)

                # Set the threshold and classify the predicted proabilities
                threshold = 0.5
                y_test_pred_class = (y_test_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the test set
                tn_test, fp_test, fn_test, tp_test, balanced_accuracy_test, precision_test, recall_test, f1_test, roc_auc_test, mcc_test, avg_precision_test = evaluate_model_performance(y_test_select, y_test_pred_class)

                # Get predictions on the training set
                y_train_pred_proba = model.predict_proba(X_train)
                # Classify the predicted proabilities
                y_train_pred_class = (y_train_pred_proba[:, 1] >= threshold).astype(int)
                # Evaluate the model performance on the training set
                tn_train, fp_train, fn_train, tp_train, balanced_accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, mcc_train, avg_precision_train = evaluate_model_performance(y_train_select, y_train_pred_class)

                end_time = time.time() 
                total_time = (end_time - start_time) / 60

                logging.info(f'Time: {total_time:.2f} minutes')
                logging.info(f'Balanced Accuracy (train) = {balanced_accuracy_train} --- Precision (train) = {precision_train} --- Recall (train)= {recall_train} --- F1 score (train)= {f1_train} --- ROC AUC score (train) = {roc_auc_train} --- MCC (train)= {mcc_train} --- PR AUC score (train)= {avg_precision_train}')
                logging.info(f'Balanced Accuracy (test) = {balanced_accuracy_test} --- Precision (test) = {precision_test} --- Recall (test)= {recall_test} --- F1 score (test)= {f1_test} --- ROC AUC score (test) = {roc_auc_test} --- MCC (test)= {mcc_test} --- PR AUC score (test)= {avg_precision_test}')
                logging.info(f'F1-score (validation): {study.best_trial.value}')

                # Construct the data column
                data_column = ', '.join(map(str, [data_dict[prefix] for prefix in x_prefixes & col_prefixes]))

                # Write values to the results file
                with open('results/'+results_filename, 'a') as resultsfile:
                    resultsfile.write(f'{data_column}\t{safety_endpoint[3:]}\t{positive_counts}({round((positive_counts/len(y_values))*100, 2)}%)\t{negative_counts} ({round((negative_counts/len(y_values))*100, 2)}%)\t'
                                      f'{algorithm}\t{num_trials}\t{study.best_trial.number}\t{tp_train}\t{fp_train}\t{fn_train}\t{tn_train}\t{tp_test}\t{fp_test}\t{fn_test}\t{tn_test}\t'
                                      f'{balanced_accuracy_train}\t{balanced_accuracy_test}\t{precision_train}\t{precision_test}\t{recall_train}\t{recall_test}\t{f1_train}\t{study.best_trial.value}\t{f1_test}\t'
                                      f'{roc_auc_train}\t{roc_auc_test}\t{mcc_train}\t{mcc_test}\t{avg_precision_train}\t{avg_precision_test}\t{total_time}\n')
     
                # Plot several model evaluation plots
                plot_model_evaluation(safety_endpoint[3:], algorithm, y_train_select, y_train_pred_proba, y_train_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=True)
                plot_model_evaluation(safety_endpoint[3:], algorithm, y_test_select, y_test_pred_proba, y_test_pred_class, show_plot=plot_results, save_path=algorithm_directory, training_data=False)
                
                if hasattr(model, 'feature_importances_'):
                    # Plot the model feature importances
                    plot_model_feature_importance(safety_endpoint[3:], algorithm, model, X_train, show_plot=plot_feature_importance, save_path=algorithm_directory)

    # Load the results TSV file
    final_df = pd.read_csv('results/'+results_filename, sep='\t')

    return final_df

# ============================= LATE DATA FUSION ==============================

def se_late_data_fusion(data_modalities, endpoint, fusion_method, save_results=False, results_filename=None):
    
    data_modalities_dict = {'cp':'Cell Painting', 'desc':'RDKit 1D descriptors', 'ecfp4':'ECFP4 fingerprints', 'mordred':'Mordred descriptors', 'pc': 'Physicochemical properties'}
    fusion_methods_list = ['average', 'weighted_average', 'voting', 'weighted_voting', 'maximal', 'weighted_maximal']

    # Define the function arguments and their expected data types and allowed values
    args_type = {'data_modalities': list, 'endpoint': Union[str, list], 'fusion_method': str, 'save_results': bool, 'results_filename': Optional[str]}
    
    # Validate argument data types
    logging.info('Validating argument data types and allowed values...')
    for argument, data_type in args_type.items():
        if not isinstance(locals()[argument], data_type):
            raise TypeError(f"'{argument}' must be a {data_type.__name__}.")
    
    # Check allowed values for the 'data_modalities' and 'fusion_method' arguments
    if isinstance(endpoint, str):
        endpoint = [endpoint] 
    for modality in data_modalities:
        if modality not in data_modalities_dict.keys():
            raise ValueError(f"'{modality}' is not a valid data modality. It must be one of {data_modalities_dict.keys()} values.")
    if fusion_method not in fusion_methods_list:
        raise ValueError(f"'{fusion_method}' is not a valid fusion method. It must be one of {fusion_methods_list} values.")
        
    # Load the safety data
    logging.info('Loading the safety data...')
    safety_data = pd.read_csv('data/5_data/PT_binary_matrix.csv')
    safety_data_columns = [col for col in safety_data.columns.tolist() if col.startswith('se')]

    # Load the TSV results file for each data modality
    results_df = []
    for modality in data_modalities:
        modality_df = pd.read_csv('data/5_data/'+modality+'_safety_results.tsv', sep='\t')
        results_df. append(modality_df)

    # Create the list of safety endpoints to be modeled
    if len(endpoint) == 1 and endpoint[0] == 'all':
        y_features = [se for se in safety_data_columns]
    else:
        y_features = ['se_'+ep for ep in endpoint]

    # Create an empty list to store metrics dictionaries
    metrics_list = []
    # Construct the data column
    data_column = ', '.join(map(str, [data_modalities_dict[modality] for modality in data_modalities]))

    # Iterate over all safety endpoints
    for safety_endpoint in safety_data_columns:

        # Check if the endpoint has to be modeled
        if safety_endpoint in y_features:

            logging.info(f'\n\n·················{safety_endpoint[3:]} ({safety_data_columns.index(safety_endpoint)+1}/{len(safety_data_columns)})·················\n')

            # Initialize two lists to store predicted probabilities and model performances for each modality
            y_test_probs = []
            model_performances = []

            # Iterate over the input data modalities
            for i in range(len(data_modalities)):
                # Select the modality and its results dataframe
                modality = data_modalities[i]
                modality_df = results_df[i]
                # Select the row corresponding to the safety endpoint results
                endpoint_results = modality_df.loc[modality_df['safety_endpoint'] == int(safety_endpoint[3:])]
                # Specify the path to the CSV file
                model_file = 'log/'+modality+'_logs/'+safety_endpoint[3:]+'/XGBoost/test_predicted_probabilities.csv'
                # Load predicted probabilities from the CSV file
                y_test_modality_probs = pd.read_csv(model_file, header=0)
                # Append the modality probabilities and the best model performance to the lists
                y_test_probs.append(y_test_modality_probs.iloc[:,1:])
                model_performances.append(endpoint_results['f1_score(validation)'].max())

            # Get the InChiKey of the test compounds
            y_test_inchis = y_test_modality_probs.iloc[:,0]
            # Select the target data corresponding to the test compounds
            test_pharmacology_data = safety_data.set_index('CPD_INCHIKEY').loc[y_test_inchis].reset_index()
            y_test_select = test_pharmacology_data.loc[:, safety_endpoint]

            if all(len(y_test_modality_probs) == len(y_test_probs[0]) for y_test_modality_probs in y_test_probs):
                # Perform the selected fusion method
                logging.info('Performing late fusion employing the selected method...')

                if fusion_method == 'average':
                    # Stack the probabilities across modalities along the third axis
                    y_test_probs = np.stack(y_test_probs, axis=2)
                    # Compute the mean probability across data modalities for each sample
                    y_test_final_probs = np.mean(y_test_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                elif fusion_method == 'weighted_average':
                    # Initialize a list to store weighted probabilities
                    y_weighted_probs = []
                    # Iterate over the probabilities and model performance of each probability
                    for probs, weight in zip(y_test_probs, model_performances):
                        # Perform element-wise multiplication
                        weighted_probs = probs * weight
                        # Append the weighted probabilities to the list
                        y_weighted_probs.append(weighted_probs)
                    # Stack the probabilities across modalities along the third axis
                    y_weighted_probs = np.stack(y_weighted_probs, axis=2)
                    # Compute the weighted mean probability across data modalities for each sample
                    y_test_final_probs = np.mean(y_weighted_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                elif fusion_method == 'voting':
                    # Initialize a list to store predicted classes
                    y_test_classes = []
                    for probs in y_test_probs:
                        # Classify the predicted probabilities
                        pred_classes = (np.array(probs)[:, 1] >= 0.5).astype(int)
                        # Append the predicted classes to the list
                        y_test_classes.append(pred_classes)
                    # Stack the classes across modalities along the third axis
                    y_test_classes = np.stack(y_test_classes, axis=1)
                    # Compute the mean class probability across data modalities for each sample
                    y_test_positive_class_probs = np.mean(y_test_classes, axis=1)

                elif fusion_method == 'weighted_voting':
                    # Initialize a list to store weighted predicted classes
                    y_weighted_classes = []
                    for probs, weight in zip(y_test_probs, model_performances):
                        # Classify the predicted probabilities
                        pred_classes = (np.array(probs)[:, 1] >= 0.5).astype(int)
                        # Perform element-wise multiplication
                        weighted_classes = pred_classes * weight
                        # Append the weighted classes to the list
                        y_weighted_classes.append(weighted_classes)
                    # Stack the probabilities across modalities along the third axis
                    y_weighted_classes = np.stack(y_weighted_classes, axis=1)
                    # Compute the weighted mean class probability across data modalities for each sample
                    y_test_positive_class_probs = np.mean(y_weighted_classes, axis=1)

                elif fusion_method == 'maximal':
                    # Stack the probabilities across modalities along the third axis
                    y_test_probs = np.stack(y_test_probs, axis=2)
                    # Compute the maximal probability across data modalities for each sample
                    y_test_final_probs = np.max(y_test_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                elif fusion_method == 'weighted_maximal':
                    # Initialize a list to store weighted probabilities
                    y_weighted_probs = []
                    # Iterate over the probabilities and model performance of each probability
                    for probs, weight in zip(y_test_probs, model_performances):
                        # Perform element-wise multiplication
                        weighted_probs = probs * weight
                        # Append the weighted probabilities to the list
                        y_weighted_probs.append(weighted_probs)
                    # Stack the probabilities across modalities along the third axis
                    y_weighted_probs = np.stack(y_weighted_probs, axis=2)
                    # Compute the weighted maximmal probability across data modalities for each sample
                    y_test_final_probs = np.max(y_weighted_probs, axis=2)
                    # Get the probability estimates of the positive class
                    y_test_positive_class_probs = y_test_final_probs[:,1]

                # Calculate the average precision from prediction scores
                avg_precision = metrics.average_precision_score(y_test_select, y_test_positive_class_probs)
                roc_auc = metrics.roc_auc_score(y_test_select, y_test_positive_class_probs)

                # Add the computed evaluation metrics to the dataframe
                metrics_list.append({'data': data_column, 'safety_endpoint': safety_endpoint[3:], 'roc_auc_score': roc_auc, 'pr_auc_score': avg_precision})
                        
            else: 
                logging.debug(f'Error: The test set\'s dimensionality (i.e., number of compounds) varies across different data modalities.')
                # Add the computed evaluation metrics to the dataframe
                metrics_list.append({'data': data_column, 'safety_endpoint': safety_endpoint[3:], 'roc_auc_score': np.nan, 'pr_auc_score': np.nan})

    # Create the DataFrame containing the metrics data
    metrics_df = pd.DataFrame(metrics_list)

    # Save the results in a TSV file
    if save_results == True: 
            if results_filename.endswith('tsv'):
                metrics_df.to_csv('results/'+results_filename, sep='\t', index=False)
            elif results_filename.endswith('csv'):
                metrics_df.to_csv('results/'+results_filename, index=False)

    # Return the metrics dataframe
    return metrics_df 