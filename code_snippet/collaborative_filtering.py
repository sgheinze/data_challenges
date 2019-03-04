# ATTENTION: This file was created for the purpose of code review - it should not be run.
# The function of the code is to take cleaned data and generate a collaborative filtering
# model using the LightFM python software package. The purpose of each function should be
# self-explanatory from the function name and associated docstrings.

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix
from scipy import sparse
import collections
from lightfm import LightFM
from lightfm import data
from lightfm import cross_validation
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from numpy import dot
from numpy.linalg import norm

def gen_collabfilt_matrix(df):
    '''Generates an interaction matrix for collaborative filtering from the raw AllTrails dataframe (df). 
    Note that each interaction will be a rating from 1-5.
    
    Returns: Interaction matrix as pandas DataFrame.'''
    
    # Initialize dictionary where each key will be the name of a hike and the associated value will itself be a dictionary 
    # where each key is a user that rated the hike and each value is the associated rating.
    hike_user_rating_dict = {} 
    
    # Grab relevant information from each hike, construct the {user_name: user_rating} dictionary, 
    # and add it to hike_user_rating_dict.
    for hike_index in range(df.shape[0]):
        hike_name = df.loc[hike_index, 'hike_name'] 
        user_names = df.loc[hike_index, 'user_names'] # This is a list
        user_ratings = df.loc[hike_index, 'user_ratings'] # This is a list
        
        # Construct {user_name: user_rating} dictionary by looping through user_names and user_ratings.
        user_rating_dict = {} 
        for user_index in range(len(user_names)): 
            user_rating_dict[user_names[user_index]] = user_ratings[user_index] 
        
        # Add user_rating_dict to hike_user_rating_dict
        hike_user_rating_dict[hike_name] = user_rating_dict
    
    return pd.DataFrame(hike_user_rating_dict, dtype='int')

def convert_to_binary(df, cutoff):
    '''Threshold an interaction matrix such that if a rating is below a certain value, the value is changed to 0;
    otherwise the value is changed to 1. A "1" indicates that there is a positive interaction between the user-item pair and
    a "0" indicates there is no interaction between the user-item pair.
    
    Returns: Thresholded interaction matrix as pandas DataFrame.'''
    
    df = df.fillna(0)
    df[df < cutoff] = 0
    df[df > cutoff] = 1
    
    return df

def lightfm_implicit_matrix(df):
    '''Utilizes the interaction matrix (df) from the previous section to create Dataset and Interactions LightFM objects, 
    which are needed to train a recommendation system model.
    
    Returns: (Dataset, Interactions) tuple, the latter as a sparse matrix (CSR)'''
    
    dataset = data.Dataset()
    dataset.fit((user for user in df.index),
                (item for item in df.columns)) # Creates mappings for users and items within LightFM
    num_users, num_items = dataset.interactions_shape() 
    interaction_list = list(df[df > 0].stack().index) # Get user-item pairs for positive interactions
    interactions, weights = dataset.build_interactions((x[0], x[1]) for x in interaction_list) # Build Interactions object within LightFM
    
    return dataset, interactions

def lightfm_train(train, num_components, num_epochs):
    '''Train a LightFM collaborative filtering model from a training set.
    
    Returns: LightFM recommendation system model.'''
    
    # Set parameters for model
    NUM_THREADS = 1
    NUM_COMPONENTS = num_components
    NUM_EPOCHS = num_epochs
    ITEM_ALPHA = 1e-6 # Recommended by LightFM

    # Let's fit a WARP model: these generally have the best performance.
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

    # Fit model
    model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
    
    return model

if __name__ == "__main__":
    hike_user_rating_matrix = gen_collabfilt_matrix(hike_data) # generate interaction matrix
    df = convert_to_binary(hike_user_rating_matrix, 2.5) # binarize interaction matrix

    # Fit model
    dataset, interactions = lightfm_implicit_matrix(interaction_matrix)
    # Create training/test set
    train, test = cross_validation.random_train_test_split(interactions, test_percentage=0.2,
                                                           random_state=np.random.RandomState(seed=1))
    #Train model
    model = lightfm_train(train, 30, 30)
    print('Great job! You trained your model!')
