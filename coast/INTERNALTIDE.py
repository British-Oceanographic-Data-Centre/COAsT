
from .COAsT import COAsT
from dask import array

class IT(COAsT):
    '''
    An object for storing Internal Tide diagnostic information
    '''


    def __init__(self, domain: COAsT, nemo: COAsT):
        self.domain_data_obj = domain
        self.nemo_data_obj = nemo
        self.zt = []
        self.zd = []
        self.depth_t = self.domain_data_obj.dataset.e3t_0.cumsum( dim='z' ).squeeze() # size: nz,my,nx


    def subset(self, points_a: array, points_b: array):
        self.dataset = super().transect(self.domain_data_obj, self.nemo_data_obj, points_a, points_b)


    def difftpt2tpt(var,dim):
        """
        Compute the Euler derivative of T-pt variable onto a T-pt.
        Input the dimension index for derivative
        """
        return  0.5*( var.roll(dim=-1, roll_coords=True)
                    - var.roll(dim=+1, roll_coords=True) )


    def get_stratification(self):
        strat = difftpt2tpt( self.votemper, dim=deptht ) / difftpt2tpt( self.depth_t, dim='z' )


    def zd(var_name='votemper', var_grid='grid_T'):
        pass


    def zt():

        # load file size
        try:
            [time_size, depth_size, lat_size, lon_size] = self.dataset['votemper'].shape
        except:
            print('I assumed that votemper existed')

        # load in background stratification data
        N2_3d = fw.variables['N2_25h'][:] # (time_counter, depth, y, x). W-pts. Surface value == 0

        # Ensure surface value is 0
        N2_3d[:,0,:,:] = 0

        # Mask at level mbathy
        print(np.shape(mbathy), lat_size,lon_size)
        indexes = [[int(mbathy[JJ,II]), JJ,II] for JJ, II in [(JJ,II) for JJ in range(lat_size) for II in range(lon_size)]]
        for index in indexes:
            #print index
            N2_3d[:,index[0],index[1],index[2]] = 999
        N2_3d[np.where(N2_3d == 999)] = np.NaN


        # initialise variables
        z_d = np.zeros((time_size,lat_size,lon_size)) # pycnocline depth
        z_t = np.zeros((time_size,lat_size,lon_size)) # pycnocline thickness


        # compute pycnocline depth, thickness and dissipation at pycnocline
        # Loop over time index to make it more simple.
    #    print 'Computing pycnocline timeseries depth, thickness and dissipation'
        for time in range(time_size):
            print('time step {} of {}'.format(time, time_size))
            N2 = N2_3d[time,:,:,:]
            eps = eps_3d[time,:,:,:]



        #    if np.shape(N2) != np.shape(z):
        #        return 'inputs variables are different shapes', np.shape(N2), np.shape(z)
            if len(np.shape(N2)) != 3:
                return 'input variable does not have the expected 3 dimensions:',  np.shape(N2)


            #
            # create list of dimension sizes to tile projection
            tile_shape = [1 for i in range(len(np.shape(N2)))]
            tile_shape[0] = np.shape(N2)[ax] # replace first dimension with the size of ax dimension (number of depth levels)
                                             # [depth_size 1 ... 1]. Tile seems to work better with new dimensions at the front
            #

            intN2  = np.nansum( N2*e3w, axis=ax) # Note that N2[k=0] = 0, so ( N2*e3w )[k=0] = 0 (is good) even though e3w[k=0] inc atm
            #zw = np.cumsum( 0, e3t, axis=ax ) # Would need to add a layer of zeros on top of this cumsum
            intzN2 = np.nansum( zw*N2*e3w, axis=ax)

            z_d[time,:,:] = intzN2 / intN2 # pycnocline depth
            z_d_tile = np.tile( z_d[time,:,:], tile_shape ).swapaxes(0,ax)

            intz2N2 = np.nansum( (zw-z_d_tile)**2 * N2 * e3w, axis=ax)
        #    intz2N2 = np.trapz( (z-z_d_tile)**2 * N2, z, axis=ax)
            z_t[time,:,:] = np.sqrt(intz2N2 / intN2)
            z_t_tile = np.tile( z_t[time,:,:], tile_shape ).swapaxes(0,ax) # pycnocline thickness


            pyc_mask = (zw >= z_d_tile-z_t_tile).astype(int) * \
                       (zw <= z_d_tile+z_t_tile).astype(int)


            ndims = np.shape(N2) # store to replace max array to shape of original array
            maxarr = np.nanmax(N2*pyc_mask, axis=ax) # store to reshape final masked field with this collapsed dimension shape
            eps_pyc[time,:,:] = np.zeros(np.shape(maxarr))*np.NaN

            Nmask = (N2 == np.tile( maxarr, tile_shape ).swapaxes(0,ax) ).astype(int) # Generate boolean mask, could use *.astype(int)
            eps_pyc[time,:,:] = np.sum( np.multiply(eps,Nmask) ,axis=ax) * np.sum( np.multiply(e3w,Nmask) ,axis=ax) # Picks out epsilon at the max N depth


        return z_t
