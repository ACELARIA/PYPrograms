# appointment.py


class Appointment:


    def __init__(self, appointment_id, patient_id, doctor_id, date):

        self.appointment_id = appointment_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id
        self.date = date



    def display_appointment(self):

        print("\nAppointment Details")
        print("Appointment ID:", self.appointment_id)
        print("Patient ID:", self.patient_id)
        print("Doctor ID:", self.doctor_id)
        print("Date:", self.date)





class AppointmentManagement:


    def __init__(self):

        self.appointments = []              # List
        self.appointment_records = {}       # Dictionary





    def book_appointment(self, doctor_system):

        appointment_id = input("Enter Appointment ID: ")

        patient_id = input("Enter Patient ID: ")

        doctor_id = input("Enter Doctor ID: ")

        date = input("Enter Appointment Date: ")



        appointment = Appointment(
            appointment_id,
            patient_id,
            doctor_id,
            date
        )



        self.appointments.append(appointment)



        self.appointment_records[appointment_id] = {

            "Patient ID": patient_id,
            "Doctor ID": doctor_id,
            "Date": date

        }



        # Adding appointment to doctor's schedule

        if doctor_id in doctor_system.doctor_records:


            doctor_system.doctor_records[doctor_id]["Schedule"].append(

                {

                    "Appointment ID": appointment_id,
                    "Patient ID": patient_id,
                    "Date": date

                }

            )


        print("\nAppointment Booked Successfully!")






    def cancel_appointment(self, doctor_system):


        appointment_id = input("Enter Appointment ID to Cancel: ")



        if appointment_id in self.appointment_records:


            doctor_id = self.appointment_records[appointment_id]["Doctor ID"]



            # Remove from appointment dictionary

            del self.appointment_records[appointment_id]



            # Remove from appointment list

            for appointment in self.appointments:


                if appointment.appointment_id == appointment_id:


                    self.appointments.remove(appointment)

                    break





            # Remove from doctor's schedule also

            if doctor_id in doctor_system.doctor_records:


                schedule = doctor_system.doctor_records[doctor_id]["Schedule"]



                for appointment in schedule:


                    if appointment["Appointment ID"] == appointment_id:


                        schedule.remove(appointment)

                        break




            print("\nAppointment Cancelled Successfully!")



        else:


            print("Appointment Not Found")







    def appointment_history(self):


        if len(self.appointments) == 0:


            print("No Appointment History Found")



        else:


            for appointment in self.appointments:


                appointment.display_appointment()