import os

with open("1.txt", "r") as f:
    data = f.readlines()

    for line in data:
        line = line.strip("\n").split(" ")
        print(line)