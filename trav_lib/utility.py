"""Classes and functions for miscellaneous tasks."""


import notebook.notebookapp
import urllib
import json
import os
import ipykernel
import shutil
import pathlib

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
        shutil.copy2(nb, new_name)
    else:
        print("Current notebook path cannot be determined.")
    return

def output_model(model,predictions,model_dir,metric,notes = ''):
    """Output serialized model, predictions, and notebook used to make model/predictions. Also
    add record to the experiment tracker."""
    
    experiment_log = model_dir / 'experiment_log.csv'
    cur_time = datetime.datetime.now()
    
    if not os.path.isfile(experiment_log):
        with open(experiment_log,'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Datetime','Model Name','Metric Score', 'Notes'])
    
    for i in range(1,1000):
        model_name = model_dir / 'model_{:03d}.pkl'.format(i)
        
        # Check if filename exists, if not write files
        if not os.path.isfile(model_name):
            with open(model_name,'wb') as f:
                pickle.dump(model,f)
                
            prediction_name = model_dir / 'predictions_{:03d}.csv'.format(i)
            predictions.to_csv(prediction_name,index=False)
            
            notebook_name = model_dir / 'notebook_{:03d}.ipynb'.format(i)
            copy_current_nb(notebook_name)
            
            with open(experiment_log, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([cur_time,f'model_{i:03d}',metric,notes])
            
            return
    print('Error: More than 1000 models in folder')
    return