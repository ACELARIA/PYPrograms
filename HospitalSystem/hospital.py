# hospital.py


class Hospital:


    def __init__(self, hospital_name):

        self.hospital_name = hospital_name



    def display_hospital(self):

        print("\n========== Hospital Details ==========")

        print("Hospital Name:", self.hospital_name)







class HospitalSystem:


    def __init__(self, hospital_name):

        self.hospital = Hospital(hospital_name)




    def show_hospital_details(self):

        self.hospital.display_hospital()







    def daily_patient_report(self, patient_system):


        print("\n========== Daily Patient Report ==========")


        total_patients = len(patient_system.patients)



        print("Total Patients Registered:", total_patients)




        if total_patients == 0:


            print("No patient records available")



        else:


            print("\nPatient List:")


            for patient in patient_system.patients:


                print("----------------------")
                print("Patient ID:", patient.patient_id)
                print("Name:", patient.name)
                print("Age:", patient.age)
                print("Disease:", patient.disease)





    def disease_report(self, patient_system):


        print("\n========== Disease Report ==========")


        if len(patient_system.unique_diseases) == 0:


            print("No diseases recorded")


        else:


            print("Unique Diseases:")


            for disease in patient_system.unique_diseases:


                print("-", disease)