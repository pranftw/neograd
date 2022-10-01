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
    self.session = None
    self.dirpath = None
    self._init_files(dirpath)
    self.model = model
  
  def new_session(self):
    '''
      Creates a new session in checkpoints.json
    '''
    self.session = self._generate_hash()
    os.mkdir(f'{self.dirpath}/{self.session}')
    return self
  
  def specify_session(self, session):
    '''
      Used to specify a particular session to use for checkpoints.json
    '''
    with open(f'{self.dirpath}/checkpoints.json', 'r') as checkpoints_fp:
      existing_checkpoints = json.load(checkpoints_fp)
      if session in existing_checkpoints:
        self.session = session
      else:
        raise ValueError(f"Invalid session {session} specified!")

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
    fname_hash = self._generate_hash()
    new_checkpoint['datetime'] = curr_time
    return new_checkpoint, fname_hash
  
  def _save(self, updated_checkpoint, params_fname_hash):
    '''
      Saves the checkpoint details onto checkpoints.json and creates a new file
        with the params of the model
    '''
    with open(f'{self.dirpath}/checkpoints.json', 'r') as checkpoints_fp:
      existing_checkpoints = json.load(checkpoints_fp)
    with open(f'{self.dirpath}/checkpoints.json', 'w') as checkpoints_fp:
      if existing_checkpoints.get(self.session) is None:
        existing_checkpoints[self.session] = {}
      existing_checkpoints[self.session][params_fname_hash] = updated_checkpoint
      json.dump(existing_checkpoints, checkpoints_fp, indent=4)
    self.model.save(f'{self.dirpath}/{self.session}/{params_fname_hash}.hkl')
  
  def load(self, params_fname, load_params=True):
    '''
      Returns the checkpoint based on the params_fname and loads the params
        onto the model if load_params is True
    '''
    params_fname_hash = params_fname.rstrip('.hkl')
    with open(f'{self.dirpath}/checkpoints.json') as checkpoints_fp:
      session = json.load(checkpoints_fp)[self.session]
      if params_fname_hash not in session.keys():
        raise ValueError(f"File {params_fname} not in current session {self.session} directory! Please specify the session using Checkpoint.specify_session")
      checkpoint = session[params_fname_hash]
    if load_params:
      self.model.load(f'{self.dirpath}/{self.session}/{params_fname}')
    return checkpoint
  
  def _init_files(self, dirpath):
    '''
      Creates a new folder at dirpath, if it doesn't exist
      A checkpoint.json file is created, if it is empty, then a new session is created
      if self.session is None, then automatically the last session is initialized as self.session
    '''
    dirpath = dirpath.rstrip("/'\'")
    try:
      os.mkdir(dirpath)
    except FileExistsError:
      pass
    self.dirpath = dirpath # All validation passed, can be assigned to self

    try:
      with open(f'{dirpath}/checkpoints.json', 'x') as checkpoints_fp: # Create checkpoints.json or do nothing if already exists
        pass
    except FileExistsError:
      pass
    
    with open(f'{dirpath}/checkpoints.json') as checkpoints_fp:
      contents = checkpoints_fp.read()
      if contents.strip()!='':
        sessions = json.loads(contents) #json.JSONDecodeError is raised if JSON file is invalid
        if self.session is None:
          self.session = list(sessions.keys())[-1]
      else: # checkpoints.json is empty
        sessions = {}
        self.new_session()
    with open(f'{dirpath}/checkpoints.json', 'w') as checkpoints_fp:
      json.dump(sessions, checkpoints_fp, indent=4)
  
  def _generate_hash(self):
    return sha256(secrets.token_hex(32).encode('utf-8')).hexdigest()