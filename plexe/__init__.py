from pathlib import Path
from typing import Optional, Union, List
from .client import PlexeAI

def build(goal: str,
          model_name: str,
          data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
          upload_id: Optional[str] = None,
          api_key: str = "",
          eval_criteria: Optional[str] = None) -> str:
    """Build a new ML model.
    
    Args:
        goal: Description of what the model should do
        model_name: Name for the model
        data_files: Optional path(s) to data file(s) to upload
        upload_id: Optional upload_id if files were already uploaded
        api_key: API key for authentication
        eval_criteria: Optional evaluation criteria
        
    Returns:
        model_version: Version ID of the created model
    """
    client = PlexeAI(api_key=api_key)
    return client.build(goal=goal, model_name=model_name,
                       data_files=data_files, upload_id=upload_id,
                       eval_criteria=eval_criteria)

async def abuild(goal: str,
                model_name: str,
                data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
                upload_id: Optional[str] = None,
                api_key: str = "",
                eval_criteria: Optional[str] = None) -> str:
    """Build a new ML model asynchronously."""
    client = PlexeAI(api_key=api_key)
    return await client.abuild(goal=goal, model_name=model_name,
                             data_files=data_files, upload_id=upload_id,
                             eval_criteria=eval_criteria)

def infer(model_name: str, model_version: str, input_data: dict, api_key: str = "") -> dict:
    """Run inference using a built model."""
    client = PlexeAI(api_key=api_key)
    return client.infer(model_name=model_name, model_version=model_version, input_data=input_data)

async def ainfer(model_name: str, model_version: str, input_data: dict, api_key: str = "") -> dict:
    """Run inference using a model asynchronously."""
    client = PlexeAI(api_key=api_key)
    return await client.ainfer(model_name=model_name, model_version=model_version, input_data=input_data)

def batch_infer(model_name: str, model_version: str, inputs: List[dict], api_key: str = "") -> List[dict]:
    """Run batch predictions."""
    client = PlexeAI(api_key=api_key)
    return client.batch_infer(model_name=model_name, model_version=model_version, inputs=inputs)

def get_status(model_name: str, model_version: str, api_key: str = "") -> dict:
    """Get status of a model build."""
    client = PlexeAI(api_key=api_key)
    return client.get_status(model_name=model_name, model_version=model_version)

async def aget_status(model_name: str, model_version: str, api_key: str = "") -> dict:
    """Get status of a model build asynchronously."""
    client = PlexeAI(api_key=api_key)
    return await client.aget_status(model_name=model_name, model_version=model_version)

def cleanup_upload(upload_id: str, api_key: str = "") -> dict:
    """Clean up uploaded files."""
    client = PlexeAI(api_key=api_key)
    return client.cleanup_upload(upload_id=upload_id)

async def acleanup_upload(upload_id: str, api_key: str = "") -> dict:
    """Clean up uploaded files asynchronously."""
    client = PlexeAI(api_key=api_key)
    return await client.acleanup_upload(upload_id=upload_id)

__all__ = ['PlexeAI', 'build', 'abuild', 'infer', 'ainfer', 
           'batch_infer', 'get_status', 'aget_status',
           'cleanup_upload', 'acleanup_upload']