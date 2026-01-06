"""
Example client script to interact with the Heart Disease Prediction API.
This script demonstrates how to call the API endpoints to get predictions.
"""
import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:5555"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_health():
    """Check if the API service is healthy."""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is the server running?")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_model_info():
    """Get information about the loaded model."""
    print_section("2. Model Information")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")


def predict_single_instance():
    """Make a prediction for a single patient."""
    print_section("3. Single Patient Prediction")

    # Sample patient data (high risk profile)
    patient_data = {
        "instances": [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        ]
    }

    print("Patient Data:")
    print(json.dumps(patient_data, indent=2))
    print("\nMaking prediction request...")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        if response.status_code == 200 and 'predictions' in result:
            pred = result['predictions'][0]
            print("\n" + "-" * 70)
            print("PREDICTION SUMMARY:")
            print(f"  Heart Disease: {'YES (1)' if pred['prediction'] == 1 else 'NO (0)'}")
            if 'confidence' in pred:
                print(f"  Confidence: {pred['confidence']:.2%}")
            print("-" * 70)
    except Exception as e:
        print(f"Error: {e}")


def predict_multiple_instances():
    """Make predictions for multiple patients."""
    print_section("4. Multiple Patients Prediction")

    # Sample data for multiple patients
    patients_data = {
        "instances": [
            {
                # Patient 1: Low risk profile
                "age": 45,
                "sex": 0,
                "cp": 0,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 170,
                "exang": 0,
                "oldpeak": 0.0,
                "slope": 0,
                "ca": 0,
                "thal": 2
            },
            {
                # Patient 2: Moderate risk profile
                "age": 55,
                "sex": 1,
                "cp": 2,
                "trestbps": 140,
                "chol": 250,
                "fbs": 1,
                "restecg": 1,
                "thalach": 140,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 1,
                "thal": 2
            },
            {
                # Patient 3: High risk profile
                "age": 70,
                "sex": 1,
                "cp": 3,
                "trestbps": 160,
                "chol": 280,
                "fbs": 1,
                "restecg": 1,
                "thalach": 130,
                "exang": 1,
                "oldpeak": 3.0,
                "slope": 2,
                "ca": 2,
                "thal": 3
            }
        ]
    }

    print(f"Number of patients: {len(patients_data['instances'])}")
    print("\nMaking batch prediction request...")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patients_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()

        if response.status_code == 200 and 'predictions' in result:
            print("\n" + "-" * 70)
            print("PREDICTION RESULTS:")
            print("-" * 70)
            for i, pred in enumerate(result['predictions'], 1):
                print(f"\nPatient {i}:")
                print(f"  Heart Disease: {'YES (1)' if pred['prediction'] == 1 else 'NO (0)'}")
                if 'confidence' in pred:
                    print(f"  Confidence: {pred['confidence']:.2%}")
                if 'probabilities' in pred:
                    print(f"  Probabilities:")
                    for cls, prob in pred['probabilities'].items():
                        print(f"    {cls}: {prob:.2%}")
            print("-" * 70)
        else:
            print(f"Response: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")


def predict_without_instances_key():
    """Make a prediction by sending data directly without 'instances' key."""
    print_section("5. Direct Prediction (without 'instances' key)")

    # Send patient data directly
    patient_data = {
        "age": 50,
        "sex": 1,
        "cp": 1,
        "trestbps": 130,
        "chol": 220,
        "fbs": 0,
        "restecg": 0,
        "thalach": 155,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }

    print("Patient Data (without 'instances' wrapper):")
    print(json.dumps(patient_data, indent=2))
    print("\nMaking prediction request...")

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function to run all examples."""
    print("\n" + "=" * 70)
    print("  HEART DISEASE PREDICTION API - CLIENT EXAMPLE")
    print("=" * 70)
    print(f"\nAPI Base URL: {BASE_URL}")

    # Check if service is healthy
    if not check_health():
        print("\nAPI is not available. Please make sure the server is running:")
        print("   python -m src.serving.app")
        return

    # Get model information
    get_model_info()

    # Run prediction examples
    predict_single_instance()
    predict_multiple_instances()
    predict_without_instances_key()

    print("\n" + "=" * 70)
    print("  EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nFor more information about the API:")
    print(f"  - Health: {BASE_URL}/health")
    print(f"  - Model Info: {BASE_URL}/model-info")
    print(f"  - Predictions: {BASE_URL}/predict (POST)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
