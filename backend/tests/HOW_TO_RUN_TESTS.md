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

## Expected Output

All tests should pass with output similar to:
```
============================================== test session starts ===============================================
platform win32 -- Python 3.13.5, pytest-8.4.1, pluggy-1.6.0
collected 9 items

tests/test_auth_service.py::TestAuthService::test_authenticate_user_success PASSED                          [ 11%]
tests/test_auth_service.py::TestAuthService::test_authenticate_user_invalid_credentials PASSED              [ 22%]
tests/test_auth_service.py::TestAuthService::test_authenticate_user_empty_credentials PASSED                [ 33%]
tests/test_aws_service.py::TestAWSStorageService::test_aws_service_initialization_success PASSED            [ 44%]
tests/test_aws_service.py::TestAWSStorageService::test_aws_service_initialization_no_credentials PASSED     [ 55%]
tests/test_aws_service.py::TestAWSStorageService::test_is_available PASSED                                  [ 66%]
tests/test_traffic_service.py::TestTrafficDataService::test_get_traffic_data_success PASSED                 [ 77%]
tests/test_traffic_service.py::TestTrafficDataService::test_get_traffic_data_with_junction_filter PASSED    [ 88%]
tests/test_traffic_service.py::TestTrafficDataService::test_get_traffic_data_empty_result PASSED            [100%]

=============================================== 9 passed in 2.62s ================================================
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
