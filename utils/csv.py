import csv

def save_loss_dict(loss_dict, path="loss_dict.csv"):
    num_epochs = len(loss_dict["train"]["loss"])

    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "set", "loss", "acc"])

        for epoch in range(num_epochs):
            for set in ["train", "valid"]:
                row = [
                    epoch + 1,
                    set,
                    loss_dict[set]["loss"][epoch],
                    loss_dict[set]["acc"][epoch],
                ]
                writer.writerow(row)
