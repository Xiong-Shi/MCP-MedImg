import os
import numpy as np
from pydicom import Dataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "test_cases", "synthetic"
)
os.makedirs(DATA_DIR, exist_ok=True)


def create_synthetic_ct(
    output_path, patient_name="TestPatient", study_desc="Lung CT", num_slices=10
):
    for i in range(num_slices):
        ds = Dataset()
        # Set file meta information (required by pydicom 3.0)
        ds.file_meta = Dataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.ImplementationClassUID = generate_uid()
        ds.file_meta.ImplementationVersionName = "MCPMEDIMG_1.0"

        ds.PatientName = patient_name
        ds.PatientID = f"SYNTH_{patient_name}"
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.StudyDescription = study_desc
        ds.SeriesDescription = "Chest CT"
        ds.Modality = "CT"
        ds.BodyPartExamined = "CHEST"
        ds.Rows = 512
        ds.Columns = 512
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        image = np.zeros((512, 512), dtype=np.int16)
        center = (256, 256)
        radius = 20 + i * 2
        y, x = np.ogrid[:512, :512]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        image[mask] = 1000
        ds.PixelData = image.tobytes()
        slice_path = os.path.join(output_path, f"slice_{i:03d}.dcm")
        ds.save_as(slice_path)
    print(f"Created {num_slices} slices in {output_path}")


print("Creating synthetic DICOM test data...")
os.makedirs(os.path.join(DATA_DIR, "case_lung_001"), exist_ok=True)
create_synthetic_ct(
    os.path.join(DATA_DIR, "case_lung_001"),
    "LungScreen001",
    "Lung Cancer Screening CT",
    10,
)
os.makedirs(os.path.join(DATA_DIR, "case_breast_001"), exist_ok=True)
create_synthetic_ct(
    os.path.join(DATA_DIR, "case_breast_001"),
    "BreastScreen001",
    "Breast Screening MG",
    1,
)
os.makedirs(os.path.join(DATA_DIR, "case_hand_001"), exist_ok=True)
create_synthetic_ct(
    os.path.join(DATA_DIR, "case_hand_001"), "BoneAge001", "Bone Age Assessment XR", 1
)
print(f"Done! Data in: {DATA_DIR}")
