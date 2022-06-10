---
title: "Process_data"
linkTitle: "Process_data"
date: 2022-06-10
description: >
  Docstrings for the Process_data class
---


### Objects

[Process_data()](#process_data)<br />
[Process_data.seasonal_decomp()](#process_dataseasonal_decomp)<br />
[Process_data.seasonal_decomposition()](#process_dataseasonal_decomposition)<br />

time series processing helper file
#### Process_data()
```python
class Process_data():
```

```
A Python class containing methods for processing time series.
```

##### Process_data.seasonal_decomp()
```python

def Process_data.seasonal_decomp(self, ts_chunk, **kwargs):
```
> <br />
> Note this is intended as an inner functions to be called by a wrapper function<br />
> Process_data.seasonal_decomposition<br />
> <br />
> This function is itself a wrapper function for statsmodel.seasonal_decompose<br />
> that accepts multiple timeseries as multiple columns (s_dim)<br />
> as an xr.DataArray: ts_chunk = ts_chunk(t_dim,s_dim).<br />
> Returns the trend, seasonal and residual componenets of the time series.<br />
> Invalid points i.e. land points, should be specified as np.nans.<br />
> <br />
> If called directly, evaluation is eager, i.e. numpy arrays are returned<br />
> for the trend, seasonal and residual components of the time series<br />
> <br />
> ts_chunk is the input array or chunk<br />
> **kwargs are the kwargs for statsmodels.seasonal_decompose<br />
> <br />
> <b>Calling with dask.delayed:</b><br />
> dask.delayed(self.seasonal_decomp, nout=3)(ts_chunk, **kwargs)<br />
> <br />
> Need statsmodels package installed<br />
> <br />
##### Process_data.seasonal_decomposition()
```python

def Process_data.seasonal_decomposition(self, time_series, num_chunks=1, **kwargs):
```
> <br />
> Dask delayed wrapper for statsmodel.seasonal_decompose that accepts multiple timeseries<br />
> <b>distributed across multiple dimensitons as an xr.DataArray:</b><br />
> time_series = time_series(t_dim,z_dim,x_dim,y_dim,...)<br />
> <br />
> Accepts land points (z,x,y) as time_series=np.nan for all time.<br />
> <br />
> num_chunks allows caller to specify the number of chunks in the delayed call, which enables<br />
> parallelism. i.e. num_chunks = 4 splits the timeserieses into 4 groups that will be decomposed<br />
> in parallel by the dask scheduler.<br />
> <br />
> **kwargs are the kwargs for statsmodels.seasonal_decompose function<br />
> <br />
> Returns coast.Gridded object containing an xarray.Dataset with the trend, seasonal, residual<br />
> timeseries components as xarray.DataArrays.<br />
> <br />
> Need statsmodels package installed<br />
> <br />
