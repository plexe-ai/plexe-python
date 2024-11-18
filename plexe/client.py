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

        self.base_url = "https://zet1g61p2k.execute-api.eu-west-2.amazonaws.com"
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

    def _get_headers(self) -> Dict[str, str]:
        """Get basic headers with API key."""
        return {
            "X-API-Key": self.api_key or "",
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
        
        # Prepare files for upload
        upload_files = []
        for f in files:
            if not f.exists():
                raise ValueError(f"File not found: {f}")
            upload_files.append(('files', (f.name, open(f, 'rb'))))

        response = self.client.post(
            f"{self.base_url}/upload",
            files=upload_files,
            headers=self._get_headers()  # Don't set Content-Type for multipart/form-data
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
            f"{self.base_url}/upload",
            files=upload_files,
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()["upload_id"]

    def build(self, goal: str, 
            data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
            data_dir: Optional[str] = None,
            steps: int = 5, 
            eval_criteria: Optional[str] = None) -> str:
        """Build a new ML model.
        
        Args:
            goal: Description of what the model should do
            data_files: Optional path(s) to data file(s) to upload
            data_dir: Optional data directory (if files already uploaded)
            steps: Number of improvement iterations
            eval_criteria: Optional evaluation criteria
            
        Returns:
            experiment_id: ID of the created experiment
        """
        if data_files is None and data_dir is None:
            raise ValueError("Either data_files or data_dir must be provided")
            
        if data_files is not None and data_dir is not None:
            raise ValueError("Cannot provide both data_files and data_dir")
            
        # Get data directory - either from upload or use provided
        if data_files is not None:
            upload_id = self.upload_files(data_files)
            data_dir = f"data/{upload_id}"
        
        # Create experiment
        response = self.client.post(
            f"{self.base_url}/experiments",
            json={
                "data_dir": data_dir,
                "goal": goal,
                "eval": eval_criteria,
                "steps": steps
            },
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()["job_id"]

    async def abuild(self, goal: str,
                    data_files: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
                    data_dir: Optional[str] = None,
                    steps: int = 5, 
                    eval_criteria: Optional[str] = None) -> str:
        """Async version of build()"""
        if data_files is None and data_dir is None:
            raise ValueError("Either data_files or data_dir must be provided")
            
        if data_files is not None and data_dir is not None:
            raise ValueError("Cannot provide both data_files and data_dir")
        
        # Get data directory - either from upload or use provided
        if data_files is not None:
            upload_id = await self.aupload_files(data_files)
            data_dir = f"data/{upload_id}"
        
        response = await self.async_client.post(
            f"{self.base_url}/experiments",
            json={
                "data_dir": data_dir,
                "goal": goal,
                "eval": eval_criteria,
                "steps": steps
            },
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()["job_id"]

    def get_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of an experiment build."""
        response = self.client.get(
            f"{self.base_url}/experiments/{experiment_id}/status",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    async def aget_status(self, experiment_id: str) -> Dict[str, Any]:
        """Async version of get_status()"""
        response = await self.async_client.get(
            f"{self.base_url}/experiments/{experiment_id}/status",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def infer(self, experiment_id: str, input_data: dict) -> Dict[str, Any]:
        """Run inference using a model."""
        response = self.client.post(
            f"{self.base_url}/models/{experiment_id}/infer",
            json=input_data,
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()

    async def ainfer(self, experiment_id: str, input_data: dict) -> Dict[str, Any]:
        """Async version of infer()"""
        response = await self.async_client.post(
            f"{self.base_url}/models/{experiment_id}/infer",
            json=input_data,
            headers=self._get_json_headers()
        )
        response.raise_for_status()
        return response.json()

    def batch_infer(self, experiment_id: str, inputs: List[dict]) -> List[Dict[str, Any]]:
        """Run batch predictions."""
        async def run_batch():
            tasks = [
                self.ainfer(experiment_id=experiment_id, input_data=x)
                for x in inputs
            ]
            return await asyncio.gather(*tasks)

        return asyncio.run(run_batch())

    def cleanup_upload(self, upload_id: str) -> Dict[str, Any]:
        """Clean up uploaded files."""
        response = self.client.delete(
            f"{self.base_url}/cleanup/{upload_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    async def acleanup_upload(self, upload_id: str) -> Dict[str, Any]:
        """Async version of cleanup_upload()"""
        response = await self.async_client.delete(
            f"{self.base_url}/cleanup/{upload_id}",
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