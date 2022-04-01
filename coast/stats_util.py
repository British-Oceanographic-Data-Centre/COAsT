"""
Python definitions used to aid with statistical calculations.

*Methods Overview*
    -> normal_distribution(): Create values for a normal distribution
    -> cumulative_distribution(): Integration udner a PDF
    -> empirical_distribution(): Estimates CDF empirically
"""

import numpy as np
import xarray as xr
import scipy

from .logging_util import error


def quadratic_spline_roots(spl):
    """
    A custom function for the roots of a quadratic spline. Cleverness found at
    https://stackoverflow.com/questions/50371298/find-maximum-minimum-of-a-1d-interpolated-function
    Used in find_maxima().

    Example usage:
    see example_scripts/tidegauge_tutorial.py
    """
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a + b) / 2), spl(b)
        t = np.roots([u + w - 2 * v, w - u, 2 * v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t * (b - a) / 2 + (b + a) / 2)
    return np.sort(roots)


def find_maxima(x, y, method="comp", **kwargs):
    """
    Finds maxima of a time series y. Returns maximum values of y (e.g heights)
    and corresponding values of x (e.g. times).
    **kwargs are dependent on method.

        Methods:
        'comp' :: Find maxima by comparison with neighbouring values.
                  Uses scipy.signal.find_peaks. **kwargs passed to this routine
                  will be passed to scipy.signal.find_peaks.
        'cubic' :: Find the maxima and minima by fitting a cubic spline and
                    finding the roots of its derivative.
                    Expect input as xr.DataArrays
        DB NOTE: Currently only the 'comp' and 'cubic' method are implemented.
                Future methods include linear interpolation.

        JP NOTE: Cubic method:
            i) has intelligent fix for NaNs,

    Example usage:
    see example_scripts/tidegauge_tutorial.py
    """

    if method == "cubic":
        if (type(x) != xr.DataArray) or (type(y) != xr.DataArray):
            msg = "With method {} require input to be type: xr.DataArray" + " not {} and {}\n Reset method as comp"
            print(msg.format(method, type(x), type(y)))
            method = "comp"

    if method == "comp":
        peaks, props = scipy.signal.find_peaks(np.copy(y), **kwargs)
        return x[peaks], y[peaks]

    if method == "cubic":
        """
        Cleverness found at
        https://stackoverflow.com/questions/50371298/find-maximum-minimum-of-a-1d-interpolated-function

        Find the extrema on a cubic spline fitted to 1d array. E.g:
        # Some data
        x_axis = xr.DataArray([ 2.14414414,  2.15270826,  2.16127238,  2.1698365 ,  2.17840062, 2.18696474,  2.19552886,  2.20409298,  2.2126571 ,  2.22122122])
        y_axis = xr.DataArray([ 0.67958442,  0.89628424,  0.78904004,  3.93404167,  6.46422317, 6.40459954,  3.80216674,  0.69641825,  0.89675386,  0.64274198])
        # Fit cubic spline
        f = scipy.interpolate.interp1d(x_axis, y_axis, kind = 'cubic')
        x_new = np.linspace(x_axis[0], x_axis[-1],100)
        cr_pts, cr_vals = stats_utils.find_maxima(x_axis, y_axis, method='cubic')
        fig = plt.subplots()
        plt.plot(x_axis, y_axis, 'r+') # The fitted spline
        plt.plot(x_new, f(x_new)) # The fitted spline
        plt.plot(cr_pts, cr_vals, 'o') # The extrema

        """
        # Remove NaNs
        nan_mask = np.isnan(y)
        if sum(nan_mask) > 0:
            print("find_maxima(): There were NaNs in timeseries")
            x = x[np.logical_not(nan_mask)]
            y = y[np.logical_not(nan_mask)]

        # Sort over time. Monotonic increasing
        y = y.sortby(x)
        x = x.sortby(x)

        # Convert x to float64 (assuming y is/similar to np.float64)
        if type(x.values[0]) == np.datetime64:  # convert to decimal sec since 1970
            x_float = ((x.values - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")).astype("float64")
            # x_float = x.values.astype('float64')
            y_float = y.values.astype("float64")
            flag_dt64 = True
        else:
            x_float = x.values.astype("float64")
            y_float = y.values.astype("float64")
            flag_dt64 = False

        if type(y.values[0]) != np.float64:
            print("find_maxima(): type(y)=", type(y))
            print("I was expecting a np.float64")

        # Do the interpolation
        f = scipy.interpolate.InterpolatedUnivariateSpline(x_float, y, k=3)
        # Find the extrema as roots
        extr_x_vals = quadratic_spline_roots(f.derivative())  # x values of extrema
        # Find the maxima roots
        extr_x_vals = np.hstack(
            [x_float[0], extr_x_vals, x_float[-1]]
        )  # add buffer points to ensure extrema are within
        ind = scipy.signal.argrelmax(f(extr_x_vals))[0]  # index that gives max(f) over extrema x locations
        max_vals = f(extr_x_vals[ind])

        # Convert back to datetime64 if appropriate
        y_out = max_vals
        if flag_dt64:
            N = len(extr_x_vals[ind])
            x_out = [
                np.datetime64("1970-01-01T00:00:00") + np.timedelta64(int(extr_x_vals[ind[i]]), "s") for i in range(N)
            ]
        else:
            x_out = extr_x_vals[ind]

        # restore xarray structure
        new_x = xr.DataArray(x_out, coords=[x_out], dims=x.dims)
        new_x.name = x.name
        new_y = xr.DataArray(y_out, coords=[x_out], dims=y.dims)
        new_y.name = y.name

        return new_x, new_y


def doodson_x0_filter(elevation, ax=0):
    """
    The Doodson X0 filter is a simple filter designed to damp out the main
    tidal frequencies. It takes hourly values, 19 values either side of the
    central one and applies a weighted average using:
              (1010010110201102112 0 2112011020110100101)/30.
    ( http://www.ntslf.org/files/acclaimdata/gloup/doodson_X0.html )

    In "Data Analaysis and Methods in Oceanography":

    "The cosine-Lanczos filter, the transform filter, and the
    Butterworth filter are often preferred to the Godin filter,
    to earlier Doodson filter, because of their superior ability
    to remove tidal period variability from oceanic signals."

    This routine can be used for any dimension input array.

    Parameters
    ----------
        elevation (ndarray) : Array of hourly elevation values.
        axis (int) : Time axis of input array. This axis must have >= 39
        elements

    Returns
    -------
        Filtered array of same rank as elevation.
    """
    if elevation.shape[ax] < 39:
        error("Doodson_XO: Ensure time axis has >=39 elements. Returning.")
        return
    # Define DOODSON XO weights
    kern = np.array(
        [
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            2,
            0,
            1,
            1,
            0,
            2,
            1,
            1,
            2,
            0,
            2,
            1,
            1,
            2,
            0,
            1,
            1,
            0,
            2,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
        ]
    )
    kern = kern / 30

    # Convolve input array with weights along the specified axis.
    filtered = np.apply_along_axis(lambda m: np.convolve(m, kern, mode=1), axis=ax, arr=elevation)

    # Pad out boundary areas with NaNs for given (arbitrary) axis.
    # DB: Is this the best way to do this?? Can put_along_axis be used instead
    filtered = filtered.swapaxes(0, ax)
    filtered[:19] = np.nan
    filtered[-19:] = np.nan
    filtered = filtered.swapaxes(0, ax)
    return filtered
