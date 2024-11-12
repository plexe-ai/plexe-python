import asyncio
from typing import Any, Dict, List, Optional, Union

import httpx

class PlexeClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self.api_key = api_key
        if not api_key:
            import os
            self.api_key = os.environ.get("PLEXE_API_KEY")
            if not self.api_key:
                raise ValueError("PLEXE_API_KEY must be provided or set as environment variable")
        
        self.base_url = "https://api.plexe.ai/v1"
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def create(self, task_description: str) -> tuple[str, int, str]:
        """Create a new ML model from a task description.
        
        Args:
            task_description: Description of what the model should do
            
        Returns:
            Tuple of (model_id, version, description)
        """
        if not task_description:
            raise ValueError("Task description must be provided")
            
        response = self.client.post(
            f"{self.base_url}/create",
            json={"description": task_description},
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        return data["model_id"], data["version"], data["description"]

    async def acreate(self, task_description: str) -> tuple[str, int, str]:
        """Async version of create()"""
        if not task_description:
            raise ValueError("Task description must be provided")
            
        response = await self.async_client.post(
            f"{self.base_url}/create",
            json={"description": task_description},
            headers=self._headers()
        )
        response.raise_for_status()
        data = response.json()
        return data["model_id"], data["version"], data["description"]

    def run(self, model_id: str, text_input: str = "", version: int = -1) -> Dict[str, Any]:
        """Run predictions using a model.
        
        Args:
            model_id: ID of the model to use
            text_input: Input text for the model
            version: Model version to use (-1 for latest)
            
        Returns:
            Dictionary containing prediction results
        """
        response = self.client.post(
            f"{self.base_url}/run",
            json={
                "model_id": model_id,
                "text": text_input,
                "version": version
            },
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()

    async def arun(self, model_id: str, text_input: str = "", version: int = -1) -> Dict[str, Any]:
        """Async version of run()"""
        response = await self.async_client.post(
            f"{self.base_url}/run",
            json={
                "model_id": model_id,
                "text": text_input,
                "version": version
            },
            headers=self._headers()
        )
        response.raise_for_status()
        return response.json()

    def batch_run(self, model_id: str, inputs: List[Dict[str, Any]], version: int = -1) -> List[Dict[str, Any]]:
        """Run batch predictions.
        
        Args:
            model_id: ID of the model to use
            inputs: List of input dictionaries
            version: Model version to use (-1 for latest)
            
        Returns:
            List of prediction results
        """
        async def run_batch():
            tasks = [
                self.arun(model_id=model_id, text_input=x.get("text", ""), version=version)
                for x in inputs
            ]
            return await asyncio.gather(*tasks)
            
        return asyncio.run(run_batch())