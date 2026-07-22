# billing.py


class Billing:


    def __init__(self, bill_id, patient_id, doctor_fee, medicine_cost):

        self.bill_id = bill_id
        self.patient_id = patient_id
        self.doctor_fee = doctor_fee
        self.medicine_cost = medicine_cost

        self.total_amount = doctor_fee + medicine_cost



    def display_bill(self):

        print("\nBill Details")
        print("Bill ID:", self.bill_id)
        print("Patient ID:", self.patient_id)
        print("Doctor Fee:", self.doctor_fee)
        print("Medicine Cost:", self.medicine_cost)
        print("Total Amount:", self.total_amount)







class BillingManagement:


    def __init__(self):

        self.bills = []              # List

        self.bill_records = {}       # Dictionary







    def generate_bill(self):


        bill_id = input("Enter Bill ID: ")

        patient_id = input("Enter Patient ID: ")


        doctor_fee = int(input("Enter Doctor Fee: "))

        medicine_cost = int(input("Enter Medicine Cost: "))




        bill = Billing(

            bill_id,
            patient_id,
            doctor_fee,
            medicine_cost

        )




        self.bills.append(bill)



        self.bill_records[bill_id] = {

            "Patient ID": patient_id,
            "Doctor Fee": doctor_fee,
            "Medicine Cost": medicine_cost,
            "Total Amount": bill.total_amount

        }



        print("\nBill Generated Successfully!")






    def display_bills(self):


        if len(self.bills) == 0:


            print("No Bills Generated")



        else:


            for bill in self.bills:


                bill.display_bill()







    def search_bill(self):


        bill_id = input("Enter Bill ID: ")



        if bill_id in self.bill_records:


            print("\nBill Found")

            print(self.bill_records[bill_id])



        else:


            print("Bill Not Found")