#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:04:17 2020
@author: Ajit Johnson Nirmal
Helper function to annotate cells based on positivity/negativity of defined markers. Users can classify
the entire data or a subset of data that has been previously phenotyped/clustered/
"""

# Library
import pandas as pd
import numpy as np


# Functions
def classify (adata, pos=None, neg=None, classify_label='passed_classify', 
              phenotype='phenotype', subclassify_phenotype=None, 
              collapse_failed=True, label="classify"):
    """
    

    Parameters
    ----------
    adata : AnnData object

    pos : list, optional
        Pass a list of markers that should be expressed in the resultant cells. The default is None.
    neg : list, optional
        Pass a list of markers that should not be expressed in the resultant cells. The default is None.
    classify_label : string, optional
        Provide a name for the calssified cells. The default is 'passed_classify'.
    subclassify_phenotype : list, optional
        If only a subset of phenotypes require to classified, pass the name of those phenotypes as a list
        through this argument. The default is None.
    phenotype : string, required
        Column name of the column containing the phenotype information. 
        This is important if `subclassify_phenotype` or `collapse_failed` arguments are used.
        The default is 'phenotype'.
    collapse_failed : bool, optional
        If set to true, the cells that were not classified based on the given criteria will be
        binned into a single category named 'failed_classify'. When False, the phenotype
        inforamation for other cells will be borrowed from the `phenotype` argument. The default is True.
    label : string, optional
        Key for the returned data, stored in `adata.obs`. The default is "classify".


    Returns
    -------
    adata : AnnData
        Updated AnnData Object.
    
    
    Example
    -------
    adata = sm.hl.classify(adata, pos=['CD3D','FOXP3'], neg=['ASMA'], subclassify_phenotype=['T cells','Regulatory T cells'])

    """
    
    # clean the input
    if isinstance(pos, str):
        pos = [pos]
    if isinstance(neg, str):
        neg = [neg]
    if isinstance(subclassify_phenotype, str):
        subclassify_phenotype = [subclassify_phenotype]
    
    
    # Create a dataFrame with the necessary inforamtion
    data = pd.DataFrame(adata.X, index= adata.obs.index, columns = adata.var.index)
    meta = pd.DataFrame(adata.obs[phenotype])
    
    # if user requests to subset a specific phenotype   
    if subclassify_phenotype is not None:
        subset_index = meta[meta[phenotype].isin(subclassify_phenotype)].index
        data = data.loc[subset_index]
        
    # Subset cells that pass the pos criteria
    if pos is not None:
        for i in pos:
            # subset data
            data = data[data[i] >= 0.5]
    
    # Subset cells that pass the neg criteria 
    if neg is not None and not data.empty:
        for j in neg:
            # subset data
            data = data[data[j] < 0.5]
    
    # cells that passed the classify criteria
    if data.empty:
        raise TypeError("No cells were found to satisfy your `classify` criteria")
    else:
        classify_idx = data.index
        classified = pd.DataFrame(np.repeat(classify_label, len(classify_idx)), index = classify_idx, columns = [label])
        
        
    # Generate a df with the classified cells
    meta = meta.merge(classified, how='outer', left_index=True, right_index=True)
    
    if collapse_failed is True:
        meta[label] = meta[label].fillna('failed_classify')
    else:
        meta = meta.fillna(method='ffill', axis = 1)
    
    # Add to Anndata
    meta = meta.reindex(adata.obs.index)
    adata.obs[label] = meta[label]
    
    # return
    return adata
        
    
    