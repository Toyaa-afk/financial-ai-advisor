import csv
def csv_cre(name,fields,rows):
    with open(name, 'w',newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
