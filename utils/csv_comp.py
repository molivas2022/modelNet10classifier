import csv

def save_loss_dict(loss_dict, path="loss_dict.csv"):
    num_epochs = len(loss_dict["train"]["our_loss"])

    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "set", "our_loss", "their_loss", "acc"])

        for epoch in range(num_epochs):
            for set in ["train", "valid"]:
                row = [
                    epoch + 1,
                    set,
                    loss_dict[set]["our_loss"][epoch],
                    loss_dict[set]["their_loss"][epoch],
                    loss_dict[set]["acc"][epoch],
                ]
                writer.writerow(row)
