'''
TODO: 
- add parameters. use calib_params.yaml just for see which params are need to be adjustable.
-
'''

from hyperopt import hp

def get_search_space()->dict:
    '''resturn the search space of hyperparameters

    Returns
    -------
    dict
        search_space
    '''    
    search_space = {
        'camera': {
            'position':hp.uniform('position',0.01,0.015),
            'yfov':hp.uniform('yfov',40.0,140)
        }
    }
    return search_space

