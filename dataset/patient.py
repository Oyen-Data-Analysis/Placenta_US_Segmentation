class Patient:
    def __init__(self, patient_id, path):
        self.patient_id = patient_id
        self.visits = [] # list of arrays of ultrasound images taken during n-th visit
        self.path = path

    
    def add_visit(self, visit):
        self.visits.append(visit)