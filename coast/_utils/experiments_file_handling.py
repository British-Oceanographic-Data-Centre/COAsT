"""Set of functions to control basic experiment file handling"""

import json
from typing import List
import numpy as np


def experiments(experiments="experiments.json"):
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
    with open(experiments, "r") as j:
        json_content = json.loads(j.read())
        try:
            exp_names = json_content["exp_names"]
        except:
            exp_names = []
        try:
            dirs = json_content["dirs"]
        except:
            dirs = []
        try:
            domains = json_content["domains"]
        except:
            domains = ""
        try:
            file_names = json_content["file_names"]
        except:
            file_names = []

        # check all non zero lengths are the same
        lengths = np.array([len(exp_names), len(dirs), len(domains), len(file_names)])
        if np.min(lengths[np.nonzero(lengths)[0]]) != np.max(lengths[np.nonzero(lengths)[0]]):
            print("Warning DIFFERENT NUMBER OF NAMES PROVIDED, CHECK JSON FILE")
    return exp_names, dirs, domains, file_names


def nemo_filename_maker(directory, year_start: int, year_stop: int, grid: str = "T") -> List:
    """Creates a list of NEMO file names from a set of standard templates.

    Args:
        directory: path to the files'
        year_start: start year
        year_stop: stop year
        grid: NEMO grid type defaults to T

    Returns: a list of possible nemo file names

    """
    # produce a list of nemo filenames
    names = []
    january = 1
    december = 13  # range is non-inclusive so we need 12 + 1

    for year in range(year_start, year_stop + 1):
        for month in range(january, december):
            new_name = f"{directory}/SENEMO_1m_{year}0101_{year}1231_grid_{grid}_{year}{month:02}-{year}{month:02}.nc"
            names.append(new_name)

    return names
