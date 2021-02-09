"""Classes and functions for miscellaneous tasks."""


import notebook.notebookapp
import urllib
import json
import os
import ipykernel
import shutil

import pickle
import os

def notebook_path():
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebook.notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
            sessions = json.load(req)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return os.path.join(srv['notebook_dir'],sess['notebook']['path'])
        except:
            pass  # There may be stale entries in the runtime directory 
    return None


def copy_current_nb(new_name):
    nb = notebook_path()
    if nb:
        new_path = os.path.join(os.path.dirname(nb), new_name+'.ipynb')
        shutil.copy2(nb, new_path)
    else:
        print("Current notebook path cannot be determined.")
    return

def output_model(model,predictions,model_dir):
    """Output serialized model, predictions, and notebook used to make model/predictions"""
    
    for i in range(1,1000):
        model_name = '{}model_{:03d}.pkl'.format(model_dir,i)
        
        # Check if filename exists, if not write files
        if not os.path.isfile(model_name):
            with open(model_name,'wb') as f:
                pickle.dump(model,f)
                
            prediction_name = '{}predictions_{:03d}.csv'.format(model_dir,i)
            predictions.to_csv(prediction_name,index=False)
            
            notebook_name = '{}notebook_{:03d}'.format(model_dir,i)
            copy_current_nb(notebook_name)
            return
    print('Error: More than 1000 models in folder')
    return