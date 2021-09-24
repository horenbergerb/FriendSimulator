import csv
import sys


def count_nonempty_lines(input_filename):
    with open(input_filename) as f:
        inreader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        return sum(line[3] != "" for line in inreader)


def parse_train_test_username_files(input_filename):
    '''Generates 3 files:
    1) input_filename +'Train.txt'
    2) input_filename + 'Test.txt'
    3) input_filename + Usernames.txt
    '''
    output_filename = ''.join(input_filename.split('.')[:-1])
    usernames = []
    total = count_nonempty_lines(input_filename)
    counter = 0
    with open(input_filename, newline='') as infile:
        inreader = csv.reader(infile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        print('Creating ' + output_filename + 'Train.txt...')
        print('Creating ' + output_filename + 'Test.txt...')
        with open(output_filename + 'Train.txt', 'w') as trainfile:
            with open(output_filename + 'Test.txt', 'w') as testfile:
                for cur_ind, row in enumerate(inreader):
                    if cur_ind == 0:
                        continue
                    if row[3] == "":
                        continue
                    if row[1] not in usernames:
                        usernames.append(row[1])
                    if counter < float(total)*.95:
                        trainfile.write('<' + row[1] + '> ' + row[3] + '\n')
                    else:
                        testfile.write('<' + row[1] + '> ' + row[3] + '\n')
                    counter += 1

    print('Creating ' + output_filename + 'Usernames.txt...')
    with open(output_filename+'Usernames.txt', 'w') as usersfile:
        for name in usernames:
            usersfile.write(name + "\n")
    print('All files created. DMs successfully parsed.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('You must provide an input filename as an argument, i.e. \'python dm_parser.py dms.csv')
    else:
        input_filename = sys.argv[1]
        parse_train_test_username_files(input_filename)
