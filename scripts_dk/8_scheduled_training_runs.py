"""
Train one network per unfinished row in a schedule spreadsheet.

Same training loop as network_training/scheduled_training_runs.py. The schedule
path is passed on the command line instead of being fixed to
intermediate/models/example_training_runs.xlsx.

Each row uses the same fields as the settings dict in
network_training/single_training_run.py, plus a finished column. Rows with
finished true are skipped. After a run, finished is set to true and the start
time, end time, and best validation loss are written back into the spreadsheet.
The experiment log is written to {project_root}/intermediate/models/experiment_log.csv.
Each run is saved in {project_root}/intermediate/models/{ID}_{name}_{val_fold}.

aug_settings, focal_class_weights, and loss_weights must be Python literals.
The output folder is {ID}_{name}_{val_fold} and training_run refuses to
overwrite an existing folder.

    python scripts_dk/8_scheduled_training_runs.py --schedule /path/to/training_runs.xlsx --project-root /path/to/lahey_training
"""

import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd


# same default project tree as scripts_dk/5_create_splits.py
project_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/lahey_training'


def as_bool(value) -> bool:
    """
    Args:
        value:  cell from the schedule spreadsheet.

    Returns:
        flag:  True or False. Text such as "false" is not treated as true.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('true', '1', 'yes'):
            return True
        if text in ('false', '0', 'no', ''):
            return False
        raise ValueError(f'Expected a boolean, got {value!r}')
    if pd.isna(value):
        return False
    return bool(value)


def as_literal(value):
    """
    Args:
        value:  cell that stores a Python list or dict as text.

    Returns:
        parsed:  the evaluated object, or the original value if it is already one.
    """
    if isinstance(value, str):
        return eval(value)
    return value


def settings_from_row(row: dict) -> dict:
    """
    Args:
        row:  one schedule row, including the finished column.

    Returns:
        settings:  configuration dict accepted by training_run.
    """
    settings = dict(row)
    del settings['finished']

    settings['ID'] = str(int(settings['ID'])).zfill(4)
    settings['name'] = str(settings['name'])
    settings['task'] = str(settings['task'])
    settings['aux_classification'] = as_bool(settings['aux_classification'])
    settings['label_folder'] = str(settings['label_folder'])
    settings['aug_settings'] = as_literal(settings['aug_settings'])
    settings['seed'] = int(settings['seed'])
    settings['val_fold'] = str(int(settings['val_fold']))
    settings['epochs'] = int(settings['epochs'])
    settings['batch_size'] = int(settings['batch_size'])
    settings['negatives'] = int(settings['negatives'])
    settings['positives_fraction'] = float(settings['positives_fraction'])
    settings['use_early_stopping'] = as_bool(settings['use_early_stopping'])
    settings['es_patience'] = int(settings['es_patience'])
    settings['model'] = str(settings['model'])
    settings['pretrained'] = as_bool(settings['pretrained'])
    settings['input_channels'] = int(settings['input_channels'])
    settings['N_classes'] = int(settings['N_classes'])
    settings['init_method'] = str(settings['init_method'])
    settings['loss_function'] = str(settings['loss_function'])
    settings['aux_loss_function'] = str(settings['aux_loss_function'])
    settings['frequency_weighting'] = as_bool(settings['frequency_weighting'])
    settings['focal_gamma'] = float(settings['focal_gamma'])
    settings['focal_class_weights'] = as_literal(settings['focal_class_weights'])
    settings['loss_weights'] = as_literal(settings['loss_weights'])
    settings['aux_weight'] = float(settings['aux_weight'])
    settings['optimizer'] = str(settings['optimizer'])
    settings['learning_rate'] = float(settings['learning_rate'])
    settings['lr_decay_patience'] = int(settings['lr_decay_patience'])
    settings['lr_decay_factor'] = float(settings['lr_decay_factor'])
    return settings


def append_experiment_log(settings: dict, models_folder: str) -> None:
    """
    Args:
        settings:  finished run, including start time, end time, and best validation loss.
        models_folder:  directory under the project root where trained networks are stored.
    """
    overview_path = os.path.join(models_folder, 'experiment_log.csv')
    if os.path.exists(overview_path):
        exp_dict = pd.read_csv(overview_path).to_dict('list')
    else:
        exp_dict = {key: [] for key in list(settings.keys())}

    for key in settings.keys():
        exp_dict[key].append(settings[key])

    pd.DataFrame.from_dict(exp_dict).to_csv(overview_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train every unfinished row in a schedule spreadsheet.')
    parser.add_argument('--schedule', required=True, help='Excel file of training runs. Unfinished rows are trained in order.')
    parser.add_argument('--project-root', default=project_root, help='Project tree. Models are written to intermediate/models under this path.')
    args = parser.parse_args()

    # training_run and ImageDataset read these paths at import time
    os.environ['BEDLUS_PROJECT_ROOT'] = os.path.abspath(args.project_root)
    if not os.path.isdir(os.environ['BEDLUS_PROJECT_ROOT']):
        raise FileNotFoundError(f'Project root not found: {os.environ["BEDLUS_PROJECT_ROOT"]}')
    from utils.config import models_folder
    from network_training.training_run import training_run
    os.makedirs(models_folder, exist_ok=True)

    schedule_path = os.path.abspath(args.schedule)
    if not os.path.isfile(schedule_path):
        raise FileNotFoundError(f'Schedule file not found: {schedule_path}')

    schedule_df = pd.read_excel(schedule_path)
    if 'finished' not in schedule_df.columns:
        raise ValueError(f'{schedule_path} has no finished column.')
    schedule_dict = schedule_df.to_dict('list')
    for column in ('start_run', 'end_run', 'best_val_loss'):
        if column not in schedule_dict:
            schedule_dict[column] = [None] * len(schedule_df)

    for idx in np.arange(len(schedule_df)):
        row = schedule_df.iloc[idx].to_dict()
        if as_bool(row['finished']):
            continue

        settings = settings_from_row(row)
        start_run, end_run, prev_best_loss = training_run(settings)

        settings['start_run'] = start_run.strftime('%m/%d/%Y %H:%M:%S')
        settings['end_run'] = end_run.strftime('%m/%d/%Y %H:%M:%S')
        settings['best_val_loss'] = prev_best_loss
        append_experiment_log(settings, models_folder)

        schedule_dict['finished'][idx] = True
        schedule_dict['start_run'][idx] = settings['start_run']
        schedule_dict['end_run'][idx] = settings['end_run']
        schedule_dict['best_val_loss'][idx] = prev_best_loss
        pd.DataFrame.from_dict(schedule_dict).to_excel(schedule_path, index=False)


if __name__ == '__main__':
    main()
