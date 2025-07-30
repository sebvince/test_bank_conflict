import csv
import sys

with open(sys.argv[1], newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    print("*************************")
    for row in reader:
        print(f"BANK Conflicts : {row['Counter_Value']}")

    print("*************************")
        
        

