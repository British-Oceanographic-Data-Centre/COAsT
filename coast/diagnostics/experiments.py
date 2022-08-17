# Set of functions to control data handling

def experiments(experiments='experiments.json'):
    import json
    import numpy as np
    with open(experiments, "r") as j:
            json_content = json.loads(j.read())
            try:
                exp_names=json_content['exp_names']
            except:
                exp_names=[]
            try:
                dirs=json_content['dirs']
            except:
                dirs=[]
            try:
                domains=json_content['domains']
            except:
                domains=''
            try:
                file_names=json_content['file_names']
            except:
                file_names=[]

            #print(lengths)
             #check all non zero lengths are the same
            lengths=np.array([len(exp_names),len(dirs),len(domains),len(file_names)])             
            if np.min(lengths[np.nonzero(lengths)[0]]) != np.max(lengths[np.nonzero(lengths)[0]]):
                print('Warning DIFFERENT NUMBER OF NAMES PROVIDED, CHECK JSON FILE')
    return  exp_names,dirs,domains,file_names      


def nemo_filenames(dpath,runtype,ystart,ystop,grid='T'):
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
                new_name="{0}/SENEMO_1m_{1}0101_{1}1231_grid_{3}_{1}{2}-{1}{2}.nc".format(dpath,YEAR,MNTH,grid)
                names.append(new_name)
                
        else:
            print('Runtype: '+ runtype + 'not coded yet, returning empty list' )
            names=[]        
        return names   



