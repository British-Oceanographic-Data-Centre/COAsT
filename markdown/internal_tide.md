---
title: "Internal_tide"
linkTitle: "Internal_tide"
date: 2022-06-10
description: >
  Docstrings for the Internal_tide class
---


### Objects

[InternalTide()](#internaltide)<br />
[InternalTide.construct_pycnocline_vars()](#internaltideconstruct_pycnocline_vars)<br />
[InternalTide.quick_plot()](#internaltidequick_plot)<br />

Internal tide class
#### InternalTide()
```python
class InternalTide(Gridded):
```

```
Object for handling and storing necessary information, methods and outputs
for calculation of internal tide diagnostics.

Herein the depth moments of stratification are used as proxies for
pycnocline depth (as the first  moment of stratification), and pycnocline
thickness  (as the 2nd moment of stratification).
This approximation improves towards the limit of a two-layer fluid.

For stratification that is not nearly two-layer, the pycnocline
thickness appears large and this method for identifying the pycnocline
depth is less reliable.

Parameters
----------
    gridded_t : xr.Dataset
        Gridded object on t-points.
    gridded_w : xr.Dataset, optional
        Gridded object on w-points.

Example basic usage:
-------------------
    # Create Internal tide diagnostics object
    IT_obj = INTERNALTIDE(gridded_t, gridded_w) # For Gridded objects on t and w-pts
    IT_obj.construct_pycnocline_vars( gridded_t, gridded_w )
    # Make maps of pycnocline thickness and depth
    IT_obj.quick_plot()
```

##### InternalTide.construct_pycnocline_vars()
```python

def InternalTide.construct_pycnocline_vars(self, gridded_t, gridded_w, strat_thres=unknown):
```
> <br />
> Computes depth moments of stratification. Under the assumption that the<br />
> stratification approximately represents a two-layer fluid, these can be<br />
> interpreted as pycnocline depths and thicknesses. They are computed on<br />
> w-points.<br />
> <br />
> 1st moment of stratification: \int z.strat dz / \int strat dz<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  In the limit of a two layer fluid this is equivalent to the<br />
> pycnocline depth, or z_d (units: metres)<br />
> <br />
> 2nd moment of stratification: \sqrt{\int (z-z_d)^2 strat dz / \int strat dz}<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  where strat = d(density)/dz<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  In the limit of a two layer fluid this is equivatlent to the<br />
> pycnocline thickness, or z_t (units: metres)<br />
> <br />
> Parameters<br />
> ----------<br />
> gridded_t : xr.Dataset<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Gridded object on t-points.<br />
> gridded_w : xr.Dataset, optional<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Gridded object on w-points.<br />
> strat_thres: float - Optional<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  limiting stratification (rho_dz < 0) to trigger masking of mixed waters<br />
> <br />
> Output<br />
> ------<br />
> self.dataset.strat_1st_mom - (t,y,x) pycnocline depth<br />
> self.dataset.strat_2nd_mom - (t,y,x) pycnocline thickness<br />
> self.dataset.strat_1st_mom_masked - (t,y,x) pycnocline depth, masked<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  in weakly stratified water beyond strat_thres<br />
> self.dataset.strat_2nd_mom_masked - (t,y,x) pycnocline thickness, masked<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  in weakly stratified water beyond strat_thres<br />
> self.dataset.mask - (t,y,x) [1/0] stratified/unstrafied<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  water column according to strat_thres not being met anywhere<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  in the column<br />
> <br />
> Returns<br />
> -------<br />
> None.<br />
> <br />
> Example Usage<br />
> -------------<br />
> # load some example data<br />
> dn_files = "./example_files/"<br />
> dn_fig = 'unit_testing/figures/'<br />
> fn_nemo_grid_t_dat = 'nemo_data_T_grid_Aug2015.nc'<br />
> fn_nemo_dom = 'COAsT_example_Nemo_domain.nc'<br />
> gridded_t = coast.Gridded(dn_files + fn_nemo_grid_t_dat,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   dn_files + fn_nemo_dom, grid_ref='t-grid')<br />
> # create an empty w-grid object, to store stratification<br />
> gridded_w = coast.Gridded( fn_domain = dn_files + fn_nemo_dom,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     grid_ref='w-grid')<br />
> <br />
> # initialise Internal Tide object<br />
> IT = coast.INTERNALTIDE(gridded_t, gridded_w)<br />
> # Construct pycnocline variables: depth and thickness<br />
> IT.construct_pycnocline_vars( gridded_t, gridded_w )<br />
> # Plot pycnocline depth and thickness<br />
> IT.quickplot()<br />
> <br />
##### InternalTide.quick_plot()
```python

def InternalTide.quick_plot(self, var=None):
```
> <br />
> Map plot for pycnocline depth and thickness variables.<br />
> <br />
> Parameters<br />
> ----------<br />
> var : xr.DataArray, optional<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Pass variable to plot. The default is None. In which case both<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  strat_1st_mom and strat_2nd_mom are plotted.<br />
> <br />
> Returns<br />
> -------<br />
> None.<br />
> <br />
> Example Usage<br />
> -------------<br />
> IT.quick_plot( 'strat_1st_mom_masked' )<br />
> <br />
