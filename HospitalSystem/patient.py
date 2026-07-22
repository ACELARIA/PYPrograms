# patient.py

class Patient:
    def __init__(self, patient_id, name, age, disease):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.disease = disease

    def display_patient(self):
        print("\nPatient Details")
        print("Patient ID:", self.patient_id)
        print("Name:", self.name)
        print("Age:", self.age)
        print("Disease:", self.disease)


class PatientManagement:

    def __init__(self):
        self.patients = []              # List
        self.patient_records = {}       # Dictionary
        self.unique_diseases = set()    # Set


    def register_patient(self):

        patient_id = input("Enter Patient ID: ")
        name = input("Enter Patient Name: ")
        age = int(input("Enter Patient Age: "))
        disease = input("Enter Disease: ")

        patient = Patient(patient_id, name, age, disease)

        self.patients.append(patient)

        self.patient_records[patient_id] = {
            "Name": name,
            "Age": age,
            "Disease": disease
        }

        self.unique_diseases.add(disease)

        print("\nPatient Registered Successfully!")


    def display_all_patients(self):

        if len(self.patients) == 0:
            print("No patients registered")
        else:
            for patient in self.patients:
                patient.display_patient()


    def search_patient(self):

        patient_id = input("Enter Patient ID to Search: ")

        if patient_id in self.patient_records:
            print("\nPatient Found")
            print(self.patient_records[patient_id])
        else:
            print("Patient Not Found")