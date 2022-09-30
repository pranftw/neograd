import os
import json
import numpy as np
import datetime as dt
import secrets
from hashlib import sha256


class Checkpoint:
  '''
    Creates and initializes files for checkpoints
    A JSON file checkpoints.json is created which contains the dict which has the tracked values
      and also the params file name at the time of adding a checkpoint, operates in append mode
    A new params file is created at each checkpoint and the JSON file is updated
  '''
  def __init__(self, model, dirpath):
    self._init_files(dirpath)
    self.model = model
    self.dirpath = dirpath.rstrip("/'\'")
  
  def add(self, **tracked):
    ''' 
      Adds a new checkpoint
    '''
    forbidden_attrs = ('datetime')
    allowed_types = (int, float, str, dict, list)
    for attr, val in tracked.items():
      if not(isinstance(val, allowed_types)):
        raise ValueError(f'Only {allowed_types} can be tracked, if using a Tensor, use data attribute and convert it into native python objects or strings!')
      if attr in forbidden_attrs:
        raise ValueError(f'Attribute {attr} must not be present in forbidden_attrs {forbidden_attrs}, as neograd uses them internally!')
    updated_checkpoint, params_fname_hash = self._update(tracked)
    self._save(updated_checkpoint, params_fname_hash)
  
  def _update(self, new_checkpoint):
    '''
      Updates the checkpoint by including the time of adding new checkpoint and
        the fname of the params of the model
    '''
    curr_time = str(dt.datetime.now())
    fname_hash = sha256(secrets.token_hex(32).encode('utf-8')).hexdigest()
    new_checkpoint['datetime'] = curr_time
    return new_checkpoint, fname_hash
  
  def _save(self, updated_checkpoint, params_fname_hash):
    '''
      Saves the checkpoint details onto checkpoints.json and creates a new file
        with the params of the model
    '''
    with open(f'{self.dirpath}/checkpoints.json', 'r') as checkpoints_fp:
      try:
        existing_checkpoints = json.load(checkpoints_fp)
      except json.JSONDecodeError as e:
        existing_checkpoints = {}
    with open(f'{self.dirpath}/checkpoints.json', 'w') as checkpoints_fp:
      existing_checkpoints[params_fname_hash] = updated_checkpoint
      json.dump(existing_checkpoints, checkpoints_fp, indent=4)
    self.model.save(f'{self.dirpath}/{params_fname_hash}.hkl')
  
  def load(self, params_fname, load_params=True):
    '''
      Returns the checkpoint based on the params_fname and loads the params
        onto the model if load_params is True
    '''
    with open(f'{self.dirpath}/checkpoints.json') as checkpoints_fp:
      checkpoint = json.load(checkpoints_fp)[params_fname.rstrip('.hkl')]
    if load_params:
      self.model.load(f'{self.dirpath}/{params_fname}')
    return checkpoint
  
  def _init_files(self, dirpath):
    dirpath = dirpath.rstrip("/'\'")
    try:
      os.mkdir(dirpath)
    except FileExistsError:
      pass
    with open(f'{dirpath}/checkpoints.json', 'a+') as checkpoints_fp: # Create checkpoints.json
      pass