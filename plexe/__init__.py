from .client import PlexeClient

def create(task_description: str, api_key: str = "") -> tuple[str, int, str]:
    """Create a new ML model."""
    client = PlexeClient(api_key=api_key)
    return client.create(task_description=task_description)

async def acreate(task_description: str, api_key: str = "") -> tuple[str, int, str]:
    """Create a new ML model asynchronously."""
    client = PlexeClient(api_key=api_key)
    return await client.acreate(task_description=task_description)

def run(model_id: str, text_input: str = "", version: int = -1, api_key: str = "") -> dict:
    """Run predictions using a model."""
    client = PlexeClient(api_key=api_key)
    return client.run(model_id=model_id, text_input=text_input, version=version)

async def arun(model_id: str, text_input: str = "", version: int = -1, api_key: str = "") -> dict:
    """Run predictions using a model asynchronously."""
    client = PlexeClient(api_key=api_key)
    return await client.arun(model_id=model_id, text_input=text_input, version=version)

def batch_run(model_id: str, inputs: list, version: int = -1, api_key: str = "") -> list:
    """Run batch predictions."""
    client = PlexeClient(api_key=api_key)
    return client.batch_run(model_id=model_id, inputs=inputs, version=version)

__all__ = ['PlexeClient', 'create', 'acreate', 'run', 'arun', 'batch_run']