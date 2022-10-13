import os
import json
import numpy as np
import datetime as dt
import secrets
from hashlib import sha256


class Checkpoint:
  '''Creates and initializes files for checkpoints
    
  A JSON file checkpoints.json is created which contains the dict which has the tracked values
  and also the params file name at the time of adding a checkpoint, operates in append mode
  A new params file is created at each checkpoint and the JSON file is updated

  Parameters:
    session (str): Current session that is in use
    dirpath (str): Directory in which checkpoints must be saved
    model (Model): Model to be checkpointed
    hash_length (int): Character length of session identifiers. Defaults to 16
  '''
  def __init__(self, model, dirpath, hash_length=16):
    '''
    Raises:
      AssertionError: if hash_length<0 and hash_length>64
    '''
    self.session = None
    self.dirpath = None
    assert hash_length>0 and hash_length<=64, 'Hash length must be between 1 and 64'
    self.hash_length = hash_length
    self._init_files(dirpath)
    self.model = model
  
  def new_session(self):
    '''Creates a new session in checkpoints.json

    Also creates a new directory for the session

    Returns:
      self
    '''
    self.session = self._generate_hash()
    os.mkdir(f'{self.dirpath}/{self.session}')
    return self
  
  def specify_session(self, session):
    '''Used to specify a particular session to use for checkpoints.json

    Args:
      session (str): The session to be used
    
    Raises:
      ValueError: If session is not already in checkpoints.json
    '''
    with open(f'{self.dirpath}/checkpoints.json', 'r') as checkpoints_fp:
      existing_checkpoints = json.load(checkpoints_fp)
      if session in existing_checkpoints:
        self.session = session
      else:
        raise ValueError(f"Invalid session {session} specified!")

  def add(self, **tracked):
    '''Adds a new checkpoint

    Args:
      **tracked: All the data that needs to be tracked in checkpoints.json
    
    Raises:
      ValueError: If forbidden_attrs ('datetime') are used as keys in tracked, because
        the same key is used by neograd to add key of the same value, which might get overwritten
      ValueError: If values in tracked aren't serializable and don't belong to builtin classes
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
    '''Updates the new_checkpoint

    Updates the checkpoint by including the datetime of adding new checkpoint
    Generates the hash that'll be used as the filename for the params file that'll be saved

    Args:
      new_checkpoint (Checkpoint): New Checkpoint to be updated
    
    Returns:
      New Checkpoint, hash that is used as fname
    '''
    curr_time = str(dt.datetime.now())
    fname_hash = self._generate_hash()
    new_checkpoint['datetime'] = curr_time
    return new_checkpoint, fname_hash
  
  def _save(self, updated_checkpoint, params_fname_hash):
    '''Saves the checkpoint

    Saves the checkpoint details onto checkpoints.json and creates a new file
    with the params of the model

    if self.session isn't already in existing checkpoints, then it creates a new dict
    and adds the checkpoint there.

    Args:
      updated_checkpoint (Checkpoint): Checkpoint that is updated
      params_fname_hash (str): Hash that is generated to be the name of filename
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
    '''Retrieves the Checkpoint

    Returns the checkpoint based on the params_fname and loads the params
    onto the model if load_params is True

    Args:
      params_fname (str): Filename to load params from
      load_params (bool): Whether params should be loaded from the file onto the model
    
    Returns:
      Checkpoint desired
    
    Raises:
      ValueError: If the current session is not present in checkpoints.py
    '''
    params_fname_hash = params_fname.rstrip('.hkl')
    with open(f'{self.dirpath}/checkpoints.json') as checkpoints_fp:
      session = json.load(checkpoints_fp).get(self.session)
      if session is None:
        raise ValueError(f"Invalid session {self.session}")
      if params_fname_hash not in session.keys():
        raise ValueError(f"File {params_fname} not in current session {self.session} directory! Please specify the session using Checkpoint.specify_session")
      checkpoint = session[params_fname_hash]
    if load_params:
      self.model.load(f'{self.dirpath}/{self.session}/{params_fname}')
    return checkpoint
  
  def _init_files(self, dirpath):
    '''Initializes files required for Checkpoint

    Creates a new folder at dirpath, if it doesn't exist
    A checkpoint.json file is created, if it is empty, then a new session is created
    if self.session is None, then automatically the last session is initialized as self.session

    Args:
      dirpath (str): Directory in which checkpoints must be saved
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
          if len(sessions)==0:
            self.new_session()
          else:
            self.session = list(sessions.keys())[-1]
      else: # checkpoints.json is empty
        sessions = {}
        self.new_session()
    with open(f'{dirpath}/checkpoints.json', 'w') as checkpoints_fp:
      json.dump(sessions, checkpoints_fp, indent=4)
  
  def _generate_hash(self):
    '''Generates 64 hex digit sha256 hash of a random number

    Returns:
      sha256 hash
    '''
    return sha256(secrets.token_hex(32).encode('utf-8')).hexdigest()[:self.hash_length]