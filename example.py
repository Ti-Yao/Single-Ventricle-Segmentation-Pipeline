from pipeline import *
data_path = 'data'
patients = sorted(glob.glob(f"{data_path}/*")) # list all the patient exams in data_path

logfile = 'error.log'

for patient in tqdm(patients): # loop through every patient exam in data path
    try:
        p = Pipeline(patient, data_path = data_path, results_path = 'results') # run the patient exam through full pipeline, saves gifs and images to results path
    except Exception as e:
        log = open(logfile, "a")
        log.write(patient + ',' + str(e)+'\n')
        log.close() # the pipeline will throw an error when it doesn't work. e.g. if there are no SAX series in the patient scan, use a log file to keep track
        
    
    
# all the information can be accessed through the p object e.g. p.volume, p.mass and p.single_dicom which is a dicom from the identified SAX series.
# use p.__dict__.keys() to see what else is stored in the p object
