from sklearn.model_selection import train_test_split
import glob

def split_data():
    all_file_paths = glob.glob("../data/*.fits")
    train_paths, test_paths = train_test_split(all_file_paths, test_size=0.2, random_state=42)
    print(f"Total FITS files found: {len(all_file_paths)}")
    print(f"Allocated for Training: {len(train_paths)}")
    print(f"Allocated for Testing: {len(test_paths)}")
    return train_paths, test_paths