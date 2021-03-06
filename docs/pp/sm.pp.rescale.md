**scimap.pp.rescale**

!!! note "Function Call"
    `scimap.pp.rescale` (
      **adata, 
      gate=None, 
      failed_markers=None, 
      method='all', 
      imageid='imageid' 
      save_fig=False**)

**Short description**

The function scales every marker between `0` and `1` such that cells that have a value less than `< 0.5` are considered negative  and cells with `> 0.5` are positive for the given marker. [`scimap.pl.gate_finder`](../../pl/sm.pl.gate_finder) can be used to identify manual gates for each marker and passed through `gate`. If manual gates are not passed, the function would attempt to rescale the data based on fitting two gaussians for each marker.  

**Parameters**

`adata` : AnnData object  

`gate` : dataframe, optional *(The default is None)*  
DataFrame with first column as markers and second column as the gate values in `log1p` scale.  
Note: If a marker is not included, the function will try to automatically identify a gate 
based on gaussian mixture modeling of the data.  
However, if a marker is included in the `gate.csv` file but no values were passed (i.e. empty), 
the marker will be simply scaled between 0-1 and will print out a warning in the console.

`failed_markers` : list, optional *(The default is None)*  
list of markers that are not expressed at all in any cell. pass in as ['CD20', 'CD3D'].  

`method` : string, optional *(The default is 'all')*  
Two available option are- 'all' or 'by_image'. In the event that multiple images were loaded in with distinct 'imageid', users have the option to scale all data together or each image independently. Please be aware of batch effects when passing 'all' with multiple images.  

`imageid` : string, optional *(The default is 'imageid')*  
Column name of the column containing the image id. The default is 'imageid'.


`save_fig` : boolian, optional *(The default is False)*  
If True, the gates identified by the GMM method will be saved in a subdirectory within your working directory.  


**Returns**

`AnnData` object with the rescaled data `adata.X`

**Example**

```
manual_gate = pd.DataFrame({'marker': ['CD3D', 'KI67'], 'gate': [7, 8]})
adata = sm.pp.rescale (adata, gate=manual_gate, failed_markers=['CD20', 'CD21'])
```
