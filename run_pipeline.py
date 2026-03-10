from pipeline import *

def main():
    data_path = './data/debug'
    log_file = 'error_log.csv'

    # Try to load existing errors to append to
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Patient', 'Error'])  # Create empty DataFrame if file doesn't exist

    patients = [pat.split('/')[-1] for pat in glob.glob(f'{data_path}/*') if 'zip' not in pat]
    print(patients)

    for patient in patients[:]:
        print(patient)
        p = Pipeline(patient, data_path=data_path)
        print(p.single_dicom.SeriesDescription)

if __name__ == "__main__":
    main()
