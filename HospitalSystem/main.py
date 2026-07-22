# main.py

from patient import PatientManagement
from doctor import DoctorManagement
from appointment import AppointmentManagement
from prescription import PrescriptionManagement
from billing import BillingManagement
from hospital import HospitalSystem


patient_system = PatientManagement()
doctor_system = DoctorManagement()
appointment_system = AppointmentManagement()
prescription_system = PrescriptionManagement()
billing_system = BillingManagement()
hospital_system = HospitalSystem("City Care Hospital")



while True:


    print("\n========== Hospital Management System ==========")

    print("1. Patient Management")
    print("2. Doctor Management")
    print("3. Appointment Management")
    print("4. Prescription Management")
    print("5. Billing Management")
    print("6. Reports")
    print("7. Exit")


    choice = int(input("Enter choice: "))





    # Patient Management

    if choice == 1:


        while True:


            print("\n========== Patient Management ==========")

            print("1. Register Patient")
            print("2. Display Patients")
            print("3. Search Patient")
            print("4. Back to Main Menu")


            patient_choice = int(input("Enter choice: "))


            if patient_choice == 1:
                patient_system.register_patient()


            elif patient_choice == 2:
                patient_system.display_all_patients()


            elif patient_choice == 3:
                patient_system.search_patient()


            elif patient_choice == 4:
                break


            else:
                print("Invalid Choice")








    # Doctor Management

    elif choice == 2:


        while True:


            print("\n========== Doctor Management ==========")

            print("1. Register Doctor")
            print("2. Display Doctors")
            print("3. Search Doctor")
            print("4. Check Doctor Availability")
            print("5. Doctor-wise Schedule")
            print("6. Back to Main Menu")


            doctor_choice = int(input("Enter choice: "))


            if doctor_choice == 1:
                doctor_system.register_doctor()


            elif doctor_choice == 2:
                doctor_system.display_all_doctors()


            elif doctor_choice == 3:
                doctor_system.search_doctor()


            elif doctor_choice == 4:
                doctor_system.check_doctor_availability()


            elif doctor_choice == 5:
                doctor_system.doctor_wise_schedule()


            elif doctor_choice == 6:
                break


            else:
                print("Invalid Choice")









    # Appointment Management

    elif choice == 3:


        while True:


            print("\n========== Appointment Management ==========")

            print("1. Book Appointment")
            print("2. Cancel Appointment")
            print("3. Appointment History")
            print("4. Back to Main Menu")


            appointment_choice = int(input("Enter choice: "))


            if appointment_choice == 1:
                appointment_system.book_appointment(doctor_system)


            elif appointment_choice == 2:
                appointment_system.cancel_appointment(doctor_system)


            elif appointment_choice == 3:
                appointment_system.appointment_history()


            elif appointment_choice == 4:
                break


            else:
                print("Invalid Choice")









    # Prescription Management

    elif choice == 4:


        while True:


            print("\n========== Prescription Management ==========")

            print("1. Generate Prescription")
            print("2. Display Prescriptions")
            print("3. Back to Main Menu")


            prescription_choice = int(input("Enter choice: "))


            if prescription_choice == 1:
                prescription_system.generate_prescription()


            elif prescription_choice == 2:
                prescription_system.display_prescriptions()


            elif prescription_choice == 3:
                break


            else:
                print("Invalid Choice")









    # Billing Management

    elif choice == 5:


        while True:


            print("\n========== Billing Management ==========")

            print("1. Generate Bill")
            print("2. Display Bills")
            print("3. Search Bill")
            print("4. Back to Main Menu")


            billing_choice = int(input("Enter choice: "))


            if billing_choice == 1:
                billing_system.generate_bill()


            elif billing_choice == 2:
                billing_system.display_bills()


            elif billing_choice == 3:
                billing_system.search_bill()


            elif billing_choice == 4:
                break


            else:
                print("Invalid Choice")










    # Reports

    elif choice == 6:


        while True:


            print("\n========== Reports ==========")

            print("1. Hospital Details")
            print("2. Daily Patient Report")
            print("3. Disease Report")
            print("4. Back to Main Menu")


            report_choice = int(input("Enter choice: "))


            if report_choice == 1:
                hospital_system.show_hospital_details()


            elif report_choice == 2:
                hospital_system.daily_patient_report(patient_system)


            elif report_choice == 3:
                hospital_system.disease_report(patient_system)


            elif report_choice == 4:
                break


            else:
                print("Invalid Choice")









    # Exit

    elif choice == 7:


        print("\nThank you for using Hospital Management System")

        break



    else:


        print("Invalid Choice")