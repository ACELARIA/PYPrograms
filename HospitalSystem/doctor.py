# doctor.py


class Doctor:

    specializations = ("Cardiology", "Orthopedic", "Dermatology")

    def __init__(self, doctor_id, name, specialization, availability):
        self.doctor_id = doctor_id
        self.name = name
        self.specialization = specialization
        self.availability = availability
        self.schedule = []


    def display_doctor(self):

        print("\nDoctor Details")
        print("Doctor ID:", self.doctor_id)
        print("Name:", self.name)
        print("Specialization:", self.specialization)
        print("Availability:", self.availability)



class DoctorManagement:


    def __init__(self):

        self.doctors = []
        self.doctor_records = {}
        self.departments = set()



    def register_doctor(self):

        doctor_id = input("Enter Doctor ID: ")
        name = input("Enter Doctor Name: ")


        print("\nAvailable Specializations:")

        for specialization in Doctor.specializations:
            print("-", specialization)


        specialization = input("Enter Specialization: ")

        availability = input("Enter Availability (Available/Not Available): ")


        doctor = Doctor(
            doctor_id,
            name,
            specialization,
            availability
        )


        self.doctors.append(doctor)


        self.doctor_records[doctor_id] = {

            "Name": name,
            "Specialization": specialization,
            "Availability": availability,
            "Schedule": []

        }


        self.departments.add(specialization)


        print("\nDoctor Registered Successfully!")



    def display_all_doctors(self):

        if len(self.doctors) == 0:

            print("No doctors registered")


        else:

            for doctor in self.doctors:

                doctor.display_doctor()



    def search_doctor(self):

        doctor_id = input("Enter Doctor ID to Search: ")


        if doctor_id in self.doctor_records:

            print("\nDoctor Found")
            print(self.doctor_records[doctor_id])


        else:

            print("Doctor Not Found")




    def check_doctor_availability(self):

        doctor_id = input("Enter Doctor ID: ")


        if doctor_id in self.doctor_records:

            print(
                "Availability:",
                self.doctor_records[doctor_id]["Availability"]
            )


        else:

            print("Doctor Not Found")





    def doctor_wise_schedule(self):

        doctor_id = input("Enter Doctor ID: ")


        if doctor_id in self.doctor_records:


            print("\nDoctor Schedule")

            schedule = self.doctor_records[doctor_id]["Schedule"]


            if len(schedule) == 0:

                print("No appointments assigned")


            else:

                for appointment in schedule:

                    print(appointment)



        else:

            print("Doctor Not Found")