---
title: "Transect"
linkTitle: "Transect"
date: 2022-06-10
description: >
  Docstrings for the Transect class
---


### Objects

[Transect()](#transect)<br />
[Transect.moving_average()](#transectmoving_average)<br />
[Transect.interpolate_slice()](#transectinterpolate_slice)<br />
[Transect.gen_z_levels()](#transectgen_z_levels)<br />
[Transect.process_transect_indices()](#transectprocess_transect_indices)<br />
[Transect.plot_transect_on_map()](#transectplot_transect_on_map)<br />
[TransectF()](#transectf)<br />
[TransectF.calc_flow_across_transect()](#transectfcalc_flow_across_transect)<br />
[TransectF._pressure_grad_fpoint()](#transectf_pressure_grad_fpoint)<br />
[TransectF.calc_geostrophic_flow()](#transectfcalc_geostrophic_flow)<br />
[TransectF.plot_normal_velocity()](#transectfplot_normal_velocity)<br />
[TransectF.plot_depth_integrated_transport()](#transectfplot_depth_integrated_transport)<br />
[TransectT()](#transectt)<br />
[TransectT.construct_pressure()](#transecttconstruct_pressure)<br />

transcet class
#### Transect()
```python
class Transect():
```

```
None
```

##### Transect.moving_average()
```python
@staticmethod
def Transect.moving_average(array_to_smooth, window=2, axis=unknown):
```
> <br />
> Returns the input array smoothed along the given axis using convolusion<br />
> <br />
##### Transect.interpolate_slice()
```python
@staticmethod
def Transect.interpolate_slice(variable_slice, depth, interpolated_depth=None):
```
> <br />
> Linearly interpolates the variable at a single time along the z_dim, which must be the<br />
> first axis.<br />
> <br />
> Parameters<br />
> ----------<br />
> variable_slice : Variable to interpolate (z_dim, r_dim)<br />
> depth : The depth at each z point for each point along the transect<br />
> interpolated_depth : (optional) desired depth profile to interpolate to. If not supplied<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  a uniform depth profile uniformaly spaced between zero and variable max depth will be used<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  with a spacing of 2 metres.<br />
> <br />
> Returns<br />
> -------<br />
> interpolated_depth_variable_slice : Interpolated variable<br />
> interpolated_depth : Interpolation depth<br />
> <br />
##### Transect.gen_z_levels()
```python
@staticmethod
def Transect.gen_z_levels(max_depth):
```
> <br />
> Generates a pre-defined 1d vertical depth coordinate,<br />
> i.e. horizontal z-level vertical coordinates up to a supplied<br />
> maximum depth, 'max_depth'<br />
> <br />
> Parameters<br />
> ----------<br />
> max_depth : int, bottom level depth<br />
> <br />
##### Transect.process_transect_indices()
```python

def Transect.process_transect_indices(self, gridded, tran_y_ind, tran_x_ind):
```
> <br />
> Get the transect indices on a specific grid<br />
> <br />
> Parameters<br />
> ----------<br />
> gridded : the model grid to define the transect on<br />
> <br />
> Return<br />
> ----------<br />
> tran_y_ind : array of y_dim indices<br />
> tran_x_ind : array of x_dim indices<br />
> <br />
##### Transect.plot_transect_on_map()
```python

def Transect.plot_transect_on_map(self):
```
> <br />
> Plot transect location on a map<br />
> <br />
> <b>Example usage:</b><br />
> --------------<br />
> tran = coast.Transect( (54,-15), (56,-12), gridded )<br />
> tran.plot_map()<br />
> <br />
#### TransectF()
```python
class TransectF(Transect):
```

```
Class defining a transect on the f-grid, which is a 3d dataset along
a linear path between a point A and a point B, with a time dimension,
a depth dimension and an along transect dimension. The model Data on f-grid
is subsetted in its entirety along these dimensions.

Note that Point A should be closer to the southern boundary of the model domain.

The user can either supply the start and end (lat,lon) coordinates of the
transect, point_A and point_B respectively, or the model y, x indices defining it.
In the latter case the user must ensure that the indices define a continuous
transect, e.g. y=[10,11,11,12], x=[5,5,6,6].
Only limited checks are performed on the suitability of the indices.

Example usage:
    point_A = (54,-15)
    point_B = (56,-12)
    transect = coast.Transect_f( gridded_f, point_A, point_B )
    or
    transect = coast.Transect_f( gridded_f, y_indices=y_ind, x_indices=x_ind )

Parameters
----------
gridded_f : GRIDDED object on the f-grid
point_a : tuple, (lat,lon)
point_b : tuple, (lat,lon)
y_indices : 1d array of model y indices defining the points of the transect
x_indices : 1d array of model x indices defining the points of the transect
```

##### TransectF.calc_flow_across_transect()
```python

def TransectF.calc_flow_across_transect(self, gridded_u, gridded_v):
```
> <br />
> Computes the flow through the transect at each segment and creates a new<br />
> dataset 'Transect_f.data_cross_tran_flow' defined on the normal velocity<br />
> points along the transect.<br />
> Transect normal velocities are calculated at each grid point and stored in<br />
> in Transect_f.data_cross_tran_flow.normal_velocities,<br />
> Depth integrated volume transport across the transect is calculated<br />
> at each transect segment and stored in Transect_f.data_cross_tran_flow.normal_transports<br />
> The latitude, longitude and the horizontal and vertical scale factors<br />
> on the normal velocity points are also stored in the dataset.<br />
> <br />
> If the time dependent cell thicknesses (e3) on the u and v grids are<br />
> present in the gridded_u and gridded_v datasets they will be used, if they<br />
> are not then the initial cell thicknesses (e3_0) will be used.<br />
> <br />
> parameters<br />
> ----------<br />
> gridded_u : GRIDDED object on the u-grid containing the i-component velocities<br />
> gridded_v : GRIDDED object on the v-gridc ontaining the j-component velocities<br />
> <br />
##### TransectF._pressure_grad_fpoint()
```python
@staticmethod
def TransectF._pressure_grad_fpoint(ds_t, ds_t_j1, ds_t_i1, ds_t_j1i1, r_ind, velocity_component):
```
> <br />
> Calculates the hydrostatic and surface pressure gradients at a set of f-points<br />
> along the transect, i.e. at a set of specific values of r_dim (but for all time and depth).<br />
> The caller must supply four datasets that contain the variables which define<br />
> the hydrostatic and surface pressure at all vertical z_levels and all time<br />
> on the t-points around the transect i.e. for a set of f-points on the transect<br />
> defined at (j+1/2, i+1/2), t-points are supplied at (j,i), (j+1,i), (j,i+1), (j+1,i+1),<br />
> corresponding to ds_T, ds_T_j1, ds_T_i1, ds_T_j1i1, respectively.<br />
> <br />
> The velocity_component defines whether u or v is normal to the transect<br />
> for the segments of the transect. A segment of transect is<br />
> defined as being r_dim to r_dim+1 where r_dim is the along transect dimension.<br />
> <br />
> Parameters<br />
> ----------<br />
> ds_t : coast.Transect_t on y=self.y_ind, x=self.x_ind<br />
> ds_t_j1 : coast.Transect_t on y=self.y_ind+1, x=self.x_ind<br />
> ds_t_i1 : coast.Transect_t on y=self.y_ind, x=self.x_ind+1<br />
> ds_t_j1i1 : coast.Transect_t on y=self.y_ind+1, x=self.x_ind+1<br />
> r_ind: 1d array, along transect indices<br />
> velocity_component : str, normal velocity at r_ind<br />
> <br />
> Returns<br />
> -------<br />
> hpg_f : DataArray with dimensions in time and depth and along transect<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  hydrostatic pressure gradient at a set of f-points along the transect<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  for all time and depth<br />
> spg_f : DataArray with dimensions in time and depth and along transect<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  surface pressure gradient at a set of f-points along the transect<br />
> <br />
##### TransectF.calc_geostrophic_flow()
```python

def TransectF.calc_geostrophic_flow(self, gridded_t, ref_density=None, config_u=config/example_nemo_grid_u.json, config_v=config/example_nemo_grid_v.json):
```
> <br />
> This method will calculate the geostrophic velocity and volume transport<br />
> (due to the geostrophic current) across the transect.<br />
> <b>4 variables are added to the Transect_f.data_cross_tran_flow dataset:</b><br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  1. normal_velocity_hpg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    (t_dim, depth_z_levels, r_dim)<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  This is the velocity due to the hydrostatic pressure gradient<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  2. normal_velocity_spg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    (t_dim, r_dim)<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  This is the velocity due to the surface pressure gradient<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  3. normal_transport_hpg  (t_dim, r_dim)<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  This is the volume transport due to the hydrostatic pressure gradient<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  4. normal_transport_AB_spg  (t_dim, r_dim<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  This is the volume transport due to the surface pressure gradient<br />
> <br />
> The implementation works by regridding from the native vertical grid to<br />
> horizontal z_levels in order to perform the horizontal gradients.<br />
> Currently the level depths are assumed fixed at their initial depths,<br />
> i.e. at time zero.<br />
> <br />
> Parameters<br />
> ----------<br />
> gridded_t : Coast<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  This is gridded model data on the t-grid for the entire domain. It<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  must contain the temperature, salinity and t-grid domain data (e1t, e2t, e3t_0).<br />
> ref_density : float, optional<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  reference density value. If None a transect mean density will be calculated<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  and used.<br />
> config_u : file<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  configuration file for u-grid object<br />
> config_v : file<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  configuration file for v-grid object<br />
> <br />
##### TransectF.plot_normal_velocity()
```python

def TransectF.plot_normal_velocity(self, time, plot_info, cmap, smoothing_window=0):
```
> <br />
> Quick plot routine of velocity across the transect AB at a specific time.<br />
> An option is provided to smooth the velocities along the transect.<br />
> NOTE: For smoothing use even integers to smooth the x and y velocities together<br />
> <br />
> <br />
> <br />
> Parameters<br />
> ---------------<br />
> time: either as integer index or actual time as a string.<br />
> plot_info: dictionary of infomation {'fig_size': value, 'title': value, 'vmin':value, 'vmax':value}<br />
> Note that if vmin and max are not set then the colourbar will be centred at zero<br />
> smoothing_window: smoothing via convolusion, larger number applies greater smoothing, recommended<br />
> to use even integers<br />
> # TODO Add cmap definition to docstring.<br />
> <br />
##### TransectF.plot_depth_integrated_transport()
```python

def TransectF.plot_depth_integrated_transport(self, time, plot_info, smoothing_window=0):
```
> <br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Quick plot routine of depth integrated transport across the transect AB at a specific time.<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  An option is provided to smooth along the transect via convolution,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  NOTE: For smoothing use even integers to smooth the x and y velocities together<br />
> <br />
> Parameters<br />
> ---------------<br />
> time: either as integer index or actual time as a string.<br />
> plot_info: dictionary of infomation {'fig_size': value, 'title': value}<br />
> smoothing_window: smoothing via convolusion, larger number applies greater smoothing.<br />
> Recommended to use even integers.<br />
> <br />
> <b>Returns:</b><br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  pyplot object<br />
> <br />
#### TransectT()
```python
class TransectT(Transect):
```

```
Class defining a transect on the t-grid, which is a 3d dataset along
a linear path between a point A and a point B, with a time dimension,
a depth dimension and an along transect dimension. The model Data on t-grid
is subsetted in its entirety along these dimensions.

Note that Point A should be closer to the southern boundary of the model domain.

The user can either supply the start and end (lat,lon) coordinates of the
transect, point_A and point_B respectively, or the model y, x indices defining it.
In the latter case the user must ensure that the indices define a continuous
transect, e.g. y=[10,11,11,12], x=[5,5,6,6].
Only limited checks are performed on the suitability of the indices.

Example usage:
    point_A = (54,-15)
    point_B = (56,-12)
    transect = coast.Transect_t( gridded_t, point_A, point_B )
    or
    transect = coast.Transect_t( gridded_t, y_indices=y_ind, x_indices=x_ind )

Parameters
----------
gridded_t : GRIDDED object on the t-grid
point_a : tuple, (lat,lon)
point_b : tuple, (lat,lon)
y_indices : 1d array of model y indices defining the points of the transect
x_indices : 1d array of model x indices defining the points of the transect
```

##### TransectT.construct_pressure()
```python

def TransectT.construct_pressure(self, ref_density=None, z_levels=None, extrapolate=False):
```
> <br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  This method is for calculating the hydrostatic and surface pressure fields<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  on horizontal levels in the vertical (z-levels). The motivation<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  is to enable the calculation of horizontal gradients; however,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  the variables can quite easily be interpolated onto the original<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  vertical grid.<br />
> <br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Requirements: The object's t-grid dataset must contain the sea surface height,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Practical Salinity and the Potential Temperature variables.<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  The GSW package is used to calculate the Absolute Pressure,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Absolute Salinity and Conservate Temperature.<br />
> <br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Three new variables (density, hydrostatic pressure, surface pressure)<br />
> <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  are created and added to the Transect_t.data dataset:</b><br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  density_zlevels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     (t_dim, depth_z_levels, r_dim)<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  pressure_h_zlevels&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  (t_dim, depth_z_levels, r_dim)<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  pressure_s&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  (t_dim, r_dim)<br />
> <br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Note that density is constructed using the EOS10<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  equation of state.<br />
> <br />
> Parameters<br />
> ----------<br />
> ref_density: float<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  reference density value, if None, then the transect mean across time,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  depth and along transect will be used.<br />
> z_levels : (optional) numpy array<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  1d array that defines the depths to interpolate the density and pressure<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  on to. If not supplied, the Transect.gen_z_levels method will be used.<br />
> extrapolate : boolean, default False<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  If true the variables are extrapolated to the deepest level, if false,<br />
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  values below the bathymetry are set to NaN<br />
> <br />
