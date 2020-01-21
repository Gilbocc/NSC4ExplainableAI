import random
import csv
import string

def random_strings(number, max_length):
    def random_string(length):
        return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

    return [random_string(random.randint(4, max_length)).capitalize() for x in range (0, number)]

def generate_data():
    num_surnames = 100000
    num_names = 4
    surnames = random_strings(num_surnames, 10)
    names = ['Pino', 'Gianni', 'Carlo', 'Zenio']
    
    people = [(names[random.randint(0, num_names - 1)], surnames[random.randint(0, num_surnames - 1)]) 
        for x in range (0, 9900)] #+ [('Giorgio', surnames[random.randint(0, num_surnames - 1)]) 
    #        for x in range (0, 100)]
    
    people = list(map(lambda x : (x[0], x[1], 0, 0), people))
    def logic(person):
        # if person[0] == 'Pino' and person[2] == 0 and person[3] == 1 : return 1
        # if person[0] == 'Pino' and person[3] == 1 : return 1
        if person[0] == 'Pino' or person[0] == 'Zenio' : return 1
        return 0

    labelled_people = list(map(lambda x : (x[0], x[1], x[2], x[3], logic(x)), people))
    random.shuffle(labelled_people)
    return labelled_people

if __name__ == '__main__':
    with open('output_2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Surname", "HasA", "HasB", "Class"])
        for i in generate_data():
            writer.writerow(i)