import csv



lista1 = [1,2,3]
lista2 = [3,4,5]
csv_columns = ['epoch', 'loss', 'val_loss']
with open("history/loss2.csv", 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    writer = csv.writer(csvfile)
    #for data in [[self.epoch_loss]]:
    #writer.writerow([self.epoch_val_loss])
    writer.writerow(csv_columns)