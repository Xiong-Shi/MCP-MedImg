import os
import requests

ORTHANC_URL = "http://localhost:8042"
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "test_cases", "synthetic"
)


def upload_dicom_folder(folder_path):
    uploaded = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    response = requests.post(
                        f"{ORTHANC_URL}/instances",
                        data=f.read(),
                        headers={"Content-Type": "application/dicom"},
                    )
                    if response.status_code == 200:
                        uploaded += 1
                        print(f"  Uploaded: {file}")
    return uploaded


print("Uploading DICOM files to Orthanc...")
for case in ["case_lung_001", "case_breast_001", "case_hand_001"]:
    case_path = os.path.join(DATA_DIR, case)
    if os.path.exists(case_path):
        print(f"\nUploading {case}...")
        count = upload_dicom_folder(case_path)
        print(f"  Total: {count} files")
print("\nVerify at: http://localhost:8042/app/explorer.html")
