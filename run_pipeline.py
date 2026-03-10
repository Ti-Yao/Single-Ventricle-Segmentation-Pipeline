import glob
import pandas as pd
from pipeline import *
import pickle
def main():
    data_path = '/workspaces/Force-ML/data/dicom_data'
    log_file = 'error_log.csv'

    # Try to load existing errors to append to
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Patient', 'Error'])  # Create empty DataFrame if file doesn't exist

    patients = [pat.split('/')[-1] for pat in glob.glob(f'{data_path}/*') if 'zip' not in pat]
    print(patients)

    for patient in ['BCH-BRETYL-1_2021'][:]:
        print(patient)
        # if not os.path.exists(f'/workspaces/Force-ML/pipeline/app/results/segs/{patient}.gif'):
            # try:
        p = Pipeline(patient, data_path=data_path)
            # except Exception as e:
            #     error_entry = pd.DataFrame([{'Patient': patient, 'Error': str(e)}])  
                
            #     # Use pd.concat() instead of append()
            #     df = pd.concat([df, error_entry], ignore_index=True)

            #     # Save updated DataFrame to CSV immediately
            #     df.to_csv(log_file, index=False)
            #     print(f"Logged error for {patient}")

if __name__ == "__main__":
    main()
