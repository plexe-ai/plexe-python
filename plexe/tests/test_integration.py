import os
import time
import pytest
import asyncio
from pathlib import Path
from plexe import PlexeAI, build, abuild, infer, ainfer, batch_infer

API_KEY = os.getenv("PLEXE_API_KEY") or ""
if not API_KEY:
    pytest.skip("PLEXE_API_KEY environment variable not set", allow_module_level=True)

TEST_MODEL_NAME = "test_prediction_model"
TEST_GOAL = "Predict the outcomes of english premier league games based on prior results using the attached dataset"

@pytest.fixture
def client():
    """Create a PlexeAI client instance for testing."""
    return PlexeAI(api_key=API_KEY)

@pytest.fixture
def sample_data_file(tmp_path):
    """Create a temporary sample data file for testing."""
    data_file = tmp_path / "test_data.csv"
    data_content = """text,sentiment
This product is amazing!,positive
I love this service,positive
Terrible experience,negative
Not worth the money,negative
Pretty good overall,positive"""
    data_file.write_text(data_content)
    return data_file

@pytest.fixture
def sample_input_data():
    """Sample input data for inference testing."""
    return {"text": "This is a great product!"}

def wait_for_model(client, model_name: str, model_version: str, timeout: int = 300):
    """Wait for model to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = client.get_status(model_name, model_version)
        if status["status"] == "completed":
            return True
        elif status["status"] == "failed":
            raise Exception(f"Model failed: {status.get('error', 'Unknown error')}")
        time.sleep(10)
    raise TimeoutError(f"Model did not complete within {timeout} seconds")

async def async_wait_for_model(client, model_name: str, model_version: str, timeout: int = 300):
    """Wait for model to be ready asynchronously."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = await client.aget_status(model_name, model_version)
        if status["status"] == "completed":
            return True
        elif status["status"] == "failed":
            raise Exception(f"Model failed: {status.get('error', 'Unknown error')}")
        await asyncio.sleep(10)
    raise TimeoutError(f"Model did not complete within {timeout} seconds")

class TestPlexeAIIntegration:
    """Integration tests for PlexeAI client."""
    
    def test_client_initialization(self):
        """Test client initialization with API key."""
        client = PlexeAI(api_key=API_KEY)
        assert client.api_key == API_KEY
        assert client.base_url == "https://api.plexe.ai/v0"

    def test_build_and_inference_flow(self, client, sample_data_file, sample_input_data):
        """Test full flow: build model with direct data files to avoid timing issues."""
        try:
            model_version = build(
                goal=TEST_GOAL,
                model_name=TEST_MODEL_NAME,
                upload_id="2d4da8f9-aaf1-4262-a36c-5e9167ca4d5b",
                api_key=API_KEY
            )
            assert isinstance(model_version, str)
            
            # Wait for model to be ready
            wait_for_model(client, TEST_MODEL_NAME, model_version)
            
            # Run inference
            result = infer(
                model_name=TEST_MODEL_NAME,
                model_version=model_version,
                input_data=sample_input_data,
                api_key=API_KEY
            )
            assert isinstance(result, dict)
            assert "prediction" in result

            # Run batch inference
            batch_inputs = [
                {"text": "Great service!"},
                {"text": "Not satisfied with the product"}
            ]
            results = batch_infer(
                model_name=TEST_MODEL_NAME,
                model_version=model_version,
                inputs=batch_inputs,
                api_key=API_KEY
            )
            assert isinstance(results, list)
            assert len(results) == len(batch_inputs)
                
        except Exception as e:
            raise e

    @pytest.mark.asyncio
    async def test_async_build_and_inference_flow(self, client, sample_data_file, sample_input_data):
        """Test full async flow: build model with direct data files to avoid timing issues."""
        try:
            # Build model asynchronously using data_files directly
            model_version = await abuild(
                goal=TEST_GOAL,
                model_name=f"{TEST_MODEL_NAME}_async",
                upload_id="2d4da8f9-aaf1-4262-a36c-5e9167ca4d5b",
                api_key=API_KEY
            )
            assert isinstance(model_version, str)
            
            # Wait for model to be ready
            await async_wait_for_model(client, f"{TEST_MODEL_NAME}_async", model_version)
            
            # Run inference asynchronously
            result = await ainfer(
                model_name=f"{TEST_MODEL_NAME}_async",
                model_version=model_version,
                input_data=sample_input_data,
                api_key=API_KEY
            )
            assert isinstance(result, dict)
            assert "prediction" in result

            # Optional batch inference test
            batch_inputs = [
                {"text": "Great service!"},
                {"text": "Not satisfied with the product"}
            ]
            results = batch_infer(
                model_name=TEST_MODEL_NAME,
                model_version=model_version,
                inputs=batch_inputs,
                api_key=API_KEY
            )
            assert isinstance(results, list)
            assert len(results) == len(batch_inputs)
                
        except Exception as e:
            raise e

    def test_file_upload_and_cleanup(self, client, sample_data_file):
        """Test file upload and cleanup."""
        upload_id = client.upload_files(sample_data_file)
        assert isinstance(upload_id, str)
        
        # Wait a bit to ensure file is processed
        time.sleep(2)
        
        cleanup_result = client.cleanup_upload(upload_id)
        assert isinstance(cleanup_result, dict)

    def test_error_handling(self, client):
        """Test error handling for invalid requests."""
        with pytest.raises(ValueError):
            build(
                goal=TEST_GOAL,
                model_name=TEST_MODEL_NAME,
                data_files=None,
                upload_id=None,
                api_key=API_KEY
            )

        with pytest.raises(ValueError):
            build(
                goal=TEST_GOAL,
                model_name=TEST_MODEL_NAME,
                data_files="nonexistent.csv",
                api_key=API_KEY
            )