import csv 
import random

numbers = [random.randint(2500,5000) for _ in range(500000)]
with open('numbers_training.csv',"w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Number","Classification"])
    for number in numbers:
        classification = 1.0 if number > 3000 else 0.0
        writer.writerow([number,classification])
print("write file successfully")