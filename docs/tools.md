# Tools


``` python
import scimap as sm
```

### Preprocessing: `pp`

`Scimap` provides a suite of tools to preprocess the data for subsequent analysis.

|                          Function                          |              Short Description               |
|:----------------------------------------------------------:|:--------------------------------------------:|
| [`sm.pp.mcmicro_to_scimap`](pp/sm.pp.mcmicro_to_scimap.md) | `mcmicro` output to scimap compatible object |
|           [`sm.pp.rescale`](../pp/sm.pp.rescale)           |    Manual/Auto gate based scaling of data    |
|                                                            |                                              |

### Tools: `tl`

|                                                                |                                                                   |
|:--------------------------------------------------------------:|:-----------------------------------------------------------------:|
|     [`sm.tl.phenotype_cells`](tl/sm.tl.phenotype_cells.md)     |          Probability distribution based cell phenotyping          |
|             [`sm.tl.cluster`](tl/sm.tl.cluster.md)             | Cluster or sub-cluster single-cells using a variety of algorithms |
|    [`sm.tl.spatial_distance`](tl/sm.tl.spatial_distance.md)    |  Computes nearest distance between all phenotypes for every cell  |
| [`sm.tl.spatial_interaction`](tl/sm.tl.spatial_interaction.md) |                  cell–cell interactions analysis                  |
|       [`sm.tl.spatial_count`](tl/sm.tl.spatial_count.md)       |        Distribution of cell-types with local neighborhood         |
|  [`sm.tl.spatial_expression`](tl/sm.tl.spatial_expression.md)  |    Distribution of spatial expression with local neighborhood     |
|   [`sm.tl.spatial_aggregate`](tl/sm.tl.spatial_aggregate.md)   |         Aggregates of cell-types with local neighborhood          |
|   [`sm.tl.spatial_pscore`](tl/sm.tl.spatial_pscore.md)   |         Scoring proximity between user defined cell types          |


### Plotting: `pl`

|                                                                |                                                                                  |
|:--------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|         [`sm.pl.gate_finder`](pl/sm.pl.gate_finder.md)         |            Overlays gating positivity on the image for manual gating             |
|        [`sm.pl.image_viewer`](../pl/sm.pl.image_viewer)        |               Opens the image with overlays on a `napari` browser                |
|    [`sm.pl.stacked_barplot`](../pl/sm.pl.stacked_barplot)    | Generate a stacked barplot from any two columns of categorical data |
| [`sm.pl.spatial_interaction`](../pl/sm.pl.spatial_interaction) |                    Heatmap of cell–cell interaction analysis                     |
|    [`sm.pl.spatial_distance`](../pl/sm.pl.spatial_distance)    | Heatmap, distribution and numeric plots to visualize distance between phenotypes |
|    [`sm.pl.spatial_pscore`](../pl/sm.pl.spatial_pscore)    | Bar plot of the derived Spatial Proximity Scores |
|                                                                |                                                                                  |

### Helper Functions: `hl`

|                                          |                                                                   |
|:----------------------------------------:|:-----------------------------------------------------------------:|
| [`sm.hl.classify`](hl/sm.hl.classify.md) | Quickly classify cells based on pos/negativity of list of markers |
| [`sm.hl.scimap_to_csv`](hl/sm.hl.scimap_to_csv.md) | Export scimap object to CSV |
| [`sm.hl.add_roi`](hl/sm.hl.add_roi.md) | Add ROI's extracted from Omero to Scimap object |
