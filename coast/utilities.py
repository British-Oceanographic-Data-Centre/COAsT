utilities


def tpt2upt(var,nx):
    """
    interpolate T-pt variable to U/V-pt variable.
    Input the axis dimension length. e.g. nx, to interpolate over
    """
    shape = np.shape(var)
    ax = shape.index(nx)
    return 0.5*( var + np.roll(var,-1, axis=ax) )

def upt2tpt(var,nx):
    """
    interpolate U-pt variable to T-pt variable.
    Input the axis dimension length. e.g. nx, to interpolate over
    """
    shape = np.shape(var)
    ax = shape.index(nx)
    return 0.5*( var + np.roll(var,+1, axis=ax) )

def vpt2upt(var,ny,nx):
    """
    interpolate v-pt variable to u-pt variable.
    Input the y-axis and x-axis dimension sizes
    """
    shape = np.shape(var)
    xax = shape.index(nx)
    yax = shape.index(ny)
    return 0.25*( var + np.roll(var,-1,axis=xax) + np.roll(var,+1,axis=yax) + np.roll(np.roll(var,+1,axis=yax),-1,axis=xax) )

def upt2vpt(var,ny,nx):
    """
    interpolate u-pt variable to v-pt variable.
    Input the y-axis and x-axis dimension indices
    """
    shape = np.shape(var)
    xax = shape.index(nx)
    yax = shape.index(ny)
    return 0.25*( var + np.roll(var,+1,axis=xax) + np.roll(var,-1,axis=yax) + np.roll(np.roll(var,-1,axis=yax),+1,axis=xax) )

def wpt2tpt(var,nz):
    """
    interpolate w-pt variable to T-pt variable.
    Input the axis dimension length. e.g. nz, to interpolate over
    """
    shape = np.shape(var)
    ax = shape.index(nz)
    return 0.5*( var + np.roll(var,-1, axis=ax) )

def diffvelpt2tpt(var,nx):
    """
    Compute the Euler derivative of U/V-pt variable onto T-pt.
    Input the dimension index for derivative
    """
    shape = np.shape(var)
    ax = shape.index(nx)
    return  var - np.roll(var,+1, axis=ax)

def difftpt2velpt(var,nx):
    """
    Compute the Euler derivative of T-pt variable onto U/V-pt.
    Input the dimension index for derivative
    """
    shape = np.shape(var)
    ax = shape.index(nx)
    return  var - np.roll(var,-1, axis=ax)

def difftpt2tpt(var,nx):
    """
    Compute the Euler derivative of T-pt variable onto a T-pt.
    Input the dimension index for derivative
    """
    shape = np.shape(var)
    ax = shape.index(nx)
    return  0.5*( np.roll(var,-1, axis=ax) - np.roll(var,+1, axis=ax) )
