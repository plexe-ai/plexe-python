import os
import asyncio
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

class PlexeAI:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        self.api_key = api_key
        if not api_key:
            self.api_key = os.environ.get("PLEXE_API_KEY")
            if not self.api_key:
                raise ValueError("PLEXE_API_KEY must be provided or set as environment variable")

        self.base_url = "https://api.plexe.ai/v0"
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    def _get_headers(self) -> Dict[str, str]:
        """Get basic headers with API key."""
        return {
            "x-api-key": self.api_key or "", 
        }

    def _get_json_headers(self) -> Dict[str, str]:
        """Get headers for JSON content."""
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        return headers

    def _ensure_list(self, data_files: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        """Convert single file path to list and ensure all paths are Path objects."""
        if isinstance(data_files, (str, Path)):
            data_files = [data_files]
        return [Path(f) for f in data_files]

    def upload_files(self, data_files: Union[str, Path, List[Union[str, Path]]]) -> str:
        """Upload data files and return upload ID."""
        files = self._ensure_list(data_files)
        
        upload_files = []
        for f in files:
            if not f.exists():
                raise ValueError(f"File not found: {f}")
            upload_files.append(('files', (f.name, open(f, 'rb'))))

        response = self.client.post(
            f"{self.base_url}/uploads",
            files=upload_files,
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()["upload_id"]

    async def aupload_files(self, data_files: Union[str, Path, List[Union[str, Path]]]) -> str:
        """Upload data files asynchronously."""
        files = self._ensure_list(data_files)
        
        upload_files = []
        for f in files:
            if not f.exists():
                raise ValueError(f"File not found: {f}")
            upload_files.append(('files', (f.name, open(f, 'rb'))))

        response = await self.async_client.post(
            f"{self.base_url}/uploads",
            files=upload_files,
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()["upload_id"]

    def build(self, 
            goal: str,
            model_name: str,
            data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
            upload_id: Optional[str] = None,
            eval_criteria: Optional[str] = None) -> str:
        """Build a new ML model.
        
        Args:
            goal: Description of what the model should do
            model_name: Name for the model
            data_files: Optional path(s) to data file(s) to upload
            upload_id: Optional upload_id if files were already uploaded
            eval_criteria: Optional evaluation criteria
            
        Returns:
            model_version: Version ID of the created model
        """
        if data_files is None and upload_id is None:
            raise ValueError("Either data_files or upload_id must be provided")
            
        if data_files is not None and upload_id is not None:
            raise ValueError("Cannot provide both data_files and upload_id")
            
        # Get upload ID - either from new upload or use provided
        if data_files is not None:
            upload_id = self.upload_files(data_files)
        
        # Create model
        response = self.client.post(
            f"{self.base_url}/models/{model_name}/create",
            json={
                "upload_id": upload_id,
                "goal": goal,
                "eval": eval_criteria
            },
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()["model_version"]

    async def abuild(self,
                    goal: str,
                    model_name: str,
                    data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
                    upload_id: Optional[str] = None,
                    eval_criteria: Optional[str] = None) -> str:
        """Async version of build()"""
        if data_files is None and upload_id is None:
            raise ValueError("Either data_files or upload_id must be provided")
            
        if data_files is not None and upload_id is not None:
            raise ValueError("Cannot provide both data_files and upload_id")
        
        # Get upload ID - either from new upload or use provided
        if data_files is not None:
            upload_id = await self.aupload_files(data_files)
        
        response = await self.async_client.post(
            f"{self.base_url}/models/{model_name}/create",
            json={
                "upload_id": upload_id,
                "goal": goal,
                "eval": eval_criteria
            },
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()["model_version"]

    def get_status(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Get status of a model build."""
        response = self.client.get(
            f"{self.base_url}/models/{model_name}/{model_version}/status",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    async def aget_status(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """Async version of get_status()"""
        response = await self.async_client.get(
            f"{self.base_url}/models/{model_name}/{model_version}/status",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def infer(self, model_name: str, model_version: str, input_data: dict) -> Dict[str, Any]:
        """Run inference using a model."""
        response = self.client.post(
            f"{self.base_url}/models/{model_name}/{model_version}/infer",
            json=input_data,
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()

    async def ainfer(self, model_name: str, model_version: str, input_data: dict) -> Dict[str, Any]:
        """Async version of infer()"""
        response = await self.async_client.post(
            f"{self.base_url}/models/{model_name}/{model_version}/infer",
            json=input_data,
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()

    def batch_infer(self, model_name: str, model_version: str, inputs: List[dict]) -> List[Dict[str, Any]]:
        """Run batch predictions."""
        async def run_batch():
            tasks = [
                self.ainfer(model_name=model_name, model_version=model_version, input_data=x)
                for x in inputs
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(run_batch())

    def cleanup_upload(self, upload_id: str) -> Dict[str, Any]:
        """Clean up uploaded files."""
        response = self.client.delete(
            f"{self.base_url}/uploads/{upload_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    async def acleanup_upload(self, upload_id: str) -> Dict[str, Any]:
        """Async version of cleanup_upload()"""
        response = await self.async_client.delete(
            f"{self.base_url}/uploads/{upload_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        asyncio.run(self.async_client.aclose())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        await self.async_client.aclose()