import csv

input_filname = "RawDMs.csv"
#the code will append 'Train.txt' and 'Test.txt' for you
output_filename = "DMsParsed"

def parse_dm_csv():
    #holds all usernames for simulating conversation later
    usernames = []

    #count number of lines
    total = None
    with open(input_filename) as f:
        total = sum(1 for line in f)

    counter = 0
    with open(input_filename, newline = '') as infile:
        inreader =  csv.reader(infile, quotechar = '"', delimiter=',', quoting=csv.QUOTE_ALL)

        with open(output_filename + 'Train.txt', 'w') as trainfile:
            with open(output_filename + 'Test.txt', 'w') as testfile:
                for l in inreader:
                    if l[3] == "":
                        continue
                    if l[1][:-5] not in usernames:
                        usernames.append(l[1][:-5])
                    if counter < float(total)*.95:
                        trainfile.write(l[1][:-5] + ": " + l[3] + "\n")
                    else:
                        testfile.write(l[1][:-5] + ": " + l[3] + "\n")
                    counter += 1
    #holds usernames for simulating conversation
    with open(output_filename+'Usernames.txt', 'w') as usersfile:
        for name in usernames:
            usersfile.write(name + "\n")
parse_dm_csv()
