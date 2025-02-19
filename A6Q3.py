class Converter:
    UNIT_CONVERSIONS = {
        'inches': 1,
        'feet': 12,  
        'yards': 36,  
        'miles': 63360,  
        'kilometers': 39370.1, 
        'meters': 39.3701, 
        'centimeters': 2.54, 
        'millimeters': 25.4 
    }

    def __init__(self, length, unit):
        self.length = length
        self.unit = unit.lower()

        if self.unit not in self.UNIT_CONVERSIONS:
            raise ValueError("Invalid unit. Please choose from inches, feet, yards, miles, kilometers, meters, centimeters, millimeters.")

    def convert(self, to_unit):
        to_unit = to_unit.lower()
        if to_unit not in self.UNIT_CONVERSIONS:
            raise ValueError("Invalid unit. Please choose from inches, feet, yards, miles, kilometers, meters, centimeters, millimeters.")
        
        length_in_inches = self.length * self.UNIT_CONVERSIONS[self.unit]
        return length_in_inches / self.UNIT_CONVERSIONS[to_unit]

    def inches(self):
        return self.convert("inches")

    def feet(self):
        return self.convert("feet")

    def yards(self):
        return self.convert("yards")

    def miles(self):
        return self.convert("miles")

    def kilometers(self):
        return self.convert("kilometers")

    def meters(self):
        return self.convert("meters")

    def centimeters(self):
        return self.convert("centimeters")

    def millimeters(self):
        return self.convert("millimeters")

if __name__ == "__main__":
    c = Converter(9, 'inches')
    
    print(f"9 inches to feet: {c.feet()}")  
    print(f"9 inches to yards: {c.yards()}")  
    print(f"9 inches to miles: {c.miles()}")  
    print(f"9 inches to centimeters: {c.centimeters()}")  
    print(f"9 inches to kilometers: {c.kilometers()}") 
    print(f"9 inches  to meters: {c.meters()}") 
    print(f"9 inches  to millimeters: {c.millimeters()}")  
