from pathlib import Path
from typing import Optional, Union, List
from .client import PlexeAI

def build(goal: str,
          data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
          data_dir: Optional[str] = None,
          api_key: str = "", 
          steps: int = 5, 
          eval_criteria: Optional[str] = None) -> str:
    """Build a new ML model.
    
    Args:
        goal: Description of what the model should do
        data_files: Optional path(s) to data file(s) to upload
        data_dir: Optional data directory (if files already uploaded)
        api_key: API key for authentication
        steps: Number of improvement iterations
        eval_criteria: Optional evaluation criteria
        
    Returns:
        experiment_id: ID of the created experiment
    """
    client = PlexeAI(api_key=api_key)
    return client.build(goal=goal, data_files=data_files, data_dir=data_dir,
                       steps=steps, eval_criteria=eval_criteria)

async def abuild(goal: str,
                data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
                data_dir: Optional[str] = None,
                api_key: str = "", 
                steps: int = 5, 
                eval_criteria: Optional[str] = None) -> str:
    """Build a new ML model asynchronously."""
    client = PlexeAI(api_key=api_key)
    return await client.abuild(goal=goal, data_files=data_files, data_dir=data_dir,
                             steps=steps, eval_criteria=eval_criteria)

def infer(experiment_id: str, input_data: dict, api_key: str = "") -> dict:
    """Run inference using a built model."""
    client = PlexeAI(api_key=api_key)
    return client.infer(experiment_id=experiment_id, input_data=input_data)

async def ainfer(experiment_id: str, input_data: dict, api_key: str = "") -> dict:
    """Run inference using a model asynchronously."""
    client = PlexeAI(api_key=api_key)
    return await client.ainfer(experiment_id=experiment_id, input_data=input_data)

def batch_infer(experiment_id: str, inputs: List[dict], api_key: str = "") -> List[dict]:
    """Run batch predictions."""
    client = PlexeAI(api_key=api_key)
    return client.batch_infer(experiment_id=experiment_id, inputs=inputs)

def get_status(experiment_id: str, api_key: str = "") -> dict:
    """Get status of an experiment build."""
    client = PlexeAI(api_key=api_key)
    return client.get_status(experiment_id=experiment_id)

async def aget_status(experiment_id: str, api_key: str = "") -> dict:
    """Get status of an experiment build asynchronously."""
    client = PlexeAI(api_key=api_key)
    return await client.aget_status(experiment_id=experiment_id)

__all__ = ['PlexeAI', 'build', 'abuild', 'infer', 'ainfer', 
           'batch_infer', 'get_status', 'aget_status']