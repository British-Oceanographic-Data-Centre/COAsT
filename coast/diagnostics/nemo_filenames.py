def nemo_filenames(dpath,runtype,ystart,ystop):
#from ..data.gridded import Gridded
#class nemo_filenames(Gridded):
#    def __init_(self,dpath,runtype,ystart,ystop):
        """
           Creates a list of NEMO file names from a set of standard templates
            Parameters
            ----------
            dpath : path to the files
            runtype : hardwired set of standard nemo filenames
            ystart : start year
            ystop : stop year
           
        """
        
        #produce a list of nemo filenames
        names=[]    
        if runtype== 'SENEMO':
         for iy in range(ystart,ystop+1):
            for im in range(1,12+1):    
                MNTH=str(im);
                if im<10:
                     MNTH='0'+ MNTH
                YEAR=str(iy)
                new_name="{0}/SENEMO_1m_{1}0101_{1}1231_grid_T_{1}{2}-{1}{2}.nc".format(dpath,YEAR,MNTH)
                names.append(new_name)
                
        else:
            print('Runtype: '+ runtype + 'not coded yet, returning empty list' )
            names=[]        
        return names   



