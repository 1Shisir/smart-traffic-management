# How to Run Tests

This guide explains how to run the tests for the Smart Traffic Management System backend.

## Prerequisites

1. Make sure you're in the backend directory:
   ```powershell
   cd c:\Users\USER\OneDrive\Desktop\smart-traffic-system\backend
   ```

2. Activate the virtual environment (if not already activated):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

## Running Tests

### Run All Tests
```powershell
python -m pytest tests/ -v
```

### Run Specific Test File
```powershell
# Run only auth service tests
python -m pytest tests/test_auth_service.py -v

# Run only traffic service tests
python -m pytest tests/test_traffic_service.py -v

# Run only AWS service tests
python -m pytest tests/test_aws_service.py -v

# Run video processing and vehicle detection tests
python -m pytest tests/test_video_processor.py -v

# Run database logging tests
python -m pytest tests/test_database_logging.py -v

# Run live streaming tests
python -m pytest tests/test_live_streaming.py -v

# Run cloud publishing tests
python -m pytest tests/test_cloud_publishing.py -v

# Run security tests
python -m pytest tests/test_security.py -v

# Run integration tests
python -m pytest tests/test_integration.py -v
```

### Run Specific Test
```powershell
# Run a specific test method
python -m pytest tests/test_auth_service.py::TestAuthService::test_authenticate_user_success -v
```

### Additional Options
```powershell
# Run tests with coverage report
python -m pytest tests/ --cov=app --cov-report=html

# Run tests in quiet mode
python -m pytest tests/ -q

# Stop on first failure
python -m pytest tests/ -x

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest tests/ -n auto
```

## Test Structure

- `conftest.py` - Test configuration and fixtures
- `test_auth_service.py` - Tests for authentication service (3 tests)
- `test_traffic_service.py` - Tests for traffic data processing service (3 tests)
- `test_aws_service.py` - Tests for AWS storage service (3 tests)
- `test_video_processor.py` - Tests for vehicle detection and video processing (10+ tests)
- `test_database_logging.py` - Tests for SQLite database operations (15+ tests)
- `test_live_streaming.py` - Tests for real-time streaming and WebSocket communication (15+ tests)
- `test_cloud_publishing.py` - Tests for AWS S3 cloud publishing (15+ tests)
- `test_security.py` - Tests for security measures and vulnerabilities (20+ tests)
- `test_integration.py` - Integration tests for complete system workflows (8+ tests)

## Expected Output

All tests should pass with output similar to:
```
============================================== test session starts ===============================================
platform win32 -- Python 3.13.5, pytest-8.4.1, pluggy-1.6.0
collected 75+ items

tests/test_auth_service.py::TestAuthService::test_authenticate_user_success PASSED                          [ 1%]
tests/test_auth_service.py::TestAuthService::test_authenticate_user_invalid_credentials PASSED              [ 2%]
tests/test_auth_service.py::TestAuthService::test_authenticate_user_empty_credentials PASSED                [ 4%]
tests/test_aws_service.py::TestAWSStorageService::test_aws_service_initialization_success PASSED            [ 5%]
tests/test_aws_service.py::TestAWSStorageService::test_aws_service_initialization_no_credentials PASSED     [ 6%]
tests/test_aws_service.py::TestAWSStorageService::test_is_available PASSED                                  [ 8%]
tests/test_traffic_service.py::TestTrafficDataService::test_get_traffic_data_success PASSED                 [ 9%]
tests/test_traffic_service.py::TestTrafficDataService::test_get_traffic_data_with_junction_filter PASSED    [10%]
tests/test_traffic_service.py::TestTrafficDataService::test_get_traffic_data_empty_result PASSED            [12%]
tests/test_video_processor.py::TestVehicleDetection::test_detect_vehicles_success PASSED                    [13%]
tests/test_video_processor.py::TestVehicleDetection::test_detect_vehicles_low_confidence PASSED             [14%]
tests/test_video_processor.py::TestVehicleDetection::test_detect_vehicles_error_handling PASSED             [16%]
tests/test_video_processor.py::TestTrafficLightStates::test_traffic_light_green_state PASSED                [17%]
tests/test_video_processor.py::TestTrafficLightStates::test_traffic_light_yellow_state PASSED               [18%]
tests/test_video_processor.py::TestTrafficLightStates::test_traffic_light_red_state PASSED                  [20%]
... (and many more tests)

=============================================== 75+ passed in 15.32s ================================================
```

## Troubleshooting

1. **Module not found errors**: Make sure you're in the backend directory and the virtual environment is activated
2. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Test failures**: Check the error output for specific issues with mocking or service logic

## Adding New Tests

To add new tests:
1. Create a new test file following the naming pattern `test_*.py`
2. Import the service you want to test
3. Use mocking to avoid external dependencies
4. Follow the existing test patterns in the current test files
