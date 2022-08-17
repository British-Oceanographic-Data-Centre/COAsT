"""Set of functions to control basic experimnet file handling"""

def experiments(experiments='experiments.json'):
    """
    Reads a json formatted files, default name is experiments.json  
    for lists of:
      experiment names (exp_names)
      directory names (dir names)
      domain file names (domains)
      file names (file_names)
   

    Parameters
    ----------
    experiments : TYPE, optional
        DESCRIPTION. The default is 'experiments.json'.

    Returns
    -------
    exp_names,dirs,domains,file_names  

    """    
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
        """
           Creates a list of NEMO file names from a set of standard templates
            Parameters
            ----------
            dpath : path to the files
            runtype : hardwired set of standard nemo filenames
            ystart : start year
            ystop : stop year
            grid (optional) NEMO grid type
            
            -------
            names :lis tof nemo file names
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

