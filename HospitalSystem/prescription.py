class Prescription:


    def __init__(self, prescription_id, patient_id, doctor_id, medicines, advice):

        self.prescription_id = prescription_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.medicines = medicines
        self.advice = advice



    def display_prescription(self):

        print("\nPrescription Details")
        print("Prescription ID:", self.prescription_id)
        print("Patient ID:", self.patient_id)
        print("Doctor ID:", self.doctor_id)
        print("Medicines:", self.medicines)
        print("Doctor Advice:", self.advice)







class PrescriptionManagement:


    def __init__(self):

        self.prescriptions = []              # List

        self.prescription_records = {}       # Dictionary






    def generate_prescription(self):


        prescription_id = input("Enter Prescription ID: ")

        patient_id = input("Enter Patient ID: ")

        doctor_id = input("Enter Doctor ID: ")



        medicines = input("Enter Medicines: ")


        advice = input("Enter Doctor Advice: ")




        prescription = Prescription(

            prescription_id,
            patient_id,
            doctor_id,
            medicines,
            advice

        )




        self.prescriptions.append(prescription)




        self.prescription_records[prescription_id] = {

            "Patient ID": patient_id,
            "Doctor ID": doctor_id,
            "Medicines": medicines,
            "Advice": advice

        }




        print("\nPrescription Generated Successfully!")







    def display_prescriptions(self):


        if len(self.prescriptions) == 0:


            print("No Prescriptions Found")



        else:


            for prescription in self.prescriptions:


                prescription.display_prescription()