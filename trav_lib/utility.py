"""Classes and functions for miscellaneous tasks."""

import os
import shutil
from pathlib import Path
import ipynbname

import pickle
import os
import datetime
import csv

def notebook_path():
    """Returns the absolute path of the notebook or None if it cannot be determined"""
    return ipynbname.path()

def copy_current_nb(new_name):
    nb = notebook_path()
    if nb:
        shutil.copy2(nb, new_name)
    else:
        print("Current notebook path cannot be determined.")
    return

def output_model(model,predictions,model_dir,cv_metric,notes = ''):
    """Output serialized model, predictions, and notebook used to make model/predictions. Also
    add record to the experiment tracker."""
    
    model_dir = Path(model_dir)

    experiment_log = model_dir / 'experiment_log.csv'
    cur_date = datetime.datetime.now().date().strftime('%Y-%m-%d')
    cur_time = datetime.datetime.now().time().strftime('%H:%M')
    
    if not os.path.isfile(experiment_log):
        with open(experiment_log,'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Date','Time','Model Name','CV Score', 'LB Score', 'Notes'])
    
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
                csvwriter.writerow([cur_date,cur_time,f'model_{i:03d}',cv_metric,'',notes])
            
            print(f'Model_{i:03d} and predictions saved.')
            return
    print('Error: More than 1000 models in folder')
    return