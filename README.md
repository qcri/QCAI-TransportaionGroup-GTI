## GTI 
GTI is a graph-based trajectory imputation approach for trajectory data completion. GTI relies on ``mutual information'' of the aggregated knowledge  of all input sparse trajectories to impute the missing data for each single one of them. GTI can act as a pre-processing step for any trajectory data management system or trajectory-based application, as it takes raw sparse trajectory data as its input and outputs dense imputed trajectory data that significantly increase the accuracy of different systems that consume trajectory data. We evaluate GTI on junction-scale as well as city-scale real datasets. In addition, GTI is used as a pre-processing step in multiple trajectory-based applications and it boosts the accuracy across these applications compared with the state-of-the-art work.  

## Input
The input is a folder containing the sparse trajectories that we want to be imputed. Each sparse trajectory is a CSV file containing the following metadata: 

<i> car_id, latitude, longitude, timestamp </i> 

We have included two sample datasets, one at data/chicago_250 which is artificially sparsified by us every 250 meters, and one at data/nyc that only has a pickup and dropoff location for each trip, collected from https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data.

## Running GTI 
GTI has 3 steps to generate an imputation. Namely generating the RoW graph, running the graph on a Go server, and then querying for imputations. Below we show a general workflow: 
```
python build_row_graph.py
```
Note that in lines 15 and 16 you need to define the location of the input dataset and the preferred output results. 

After that is finished, make sure that centroids.txt, edges.txt, edges_go.txt are created in the output folder. 

In one terminal run the go server listening at some port (i.e 3333): 
```
go run routing_distance.go 3333
```

In another terminal run the interpolation sending requests at the same port (3333): 
```
python interpolate_and_refine.py 3333
```
Note that in lines 117, 119, 122, 125, and 126 you can manually change the location of your input/output trajectories. 
## Output
The output of interpolate_and_refine.py will be a folder in the output folder named GTI where the GTI imputation and the corresponding timestamps are stored and another GTI_refined folder where the imputed trajectories have been refined as per the best-fit algorithm described in the paper. 
