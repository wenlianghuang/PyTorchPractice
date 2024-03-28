import csv 
import random

numbers = [random.randint(500,8000) for _ in range(500000)]
with open('numbers_training.csv',"w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Number","Classification"])
    for number in numbers:
        #classification = 1.0 if number > 3000 else 0.0
        #writer.writerow([number,classification])
        if number > 500 and number <=1000:
            classification = 1
        elif number > 1000 and number <= 1500:
            classification = 2
        elif number > 1500 and number <= 2000:
            classification = 3
        elif number > 2000 and number <= 2500:
            classification = 4
        elif number > 2500 and number <= 3000:
            classification = 5
        elif number > 3000 and number <= 3500:
            classification = 6
        elif number > 3500 and number <= 4000:
            classification = 7
        elif number > 4000 and number <= 4500:
            classification = 8
        elif number > 4500 and number <= 5000:
            classification = 9
        elif number > 5000 and number <= 5500:
            classification = 10
        else:
            classification = 11
        writer.writerow([number,classification])
print("write file successfully")