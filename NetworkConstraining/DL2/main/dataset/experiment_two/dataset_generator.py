import random
import csv

FEATURES = [str(x) for x in range(1, 31)]
TARGET = "Class"
NUM_ROWS = 1000

def logic(row):
    if row[10] == 1 and row[15] == 0 and row[25] == 1 : return 1
    return 0

def generate_data():
    data = [[random.randint(0, 1) for y in FEATURES] for _ in range (0, NUM_ROWS)]
    labelled_data = list(map(lambda x : x + [logic(x)], data))
    random.shuffle(labelled_data)
    return labelled_data

if __name__ == '__main__':
    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(FEATURES + [TARGET])
        for i in generate_data():
            writer.writerow(i)