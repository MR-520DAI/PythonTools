import random

# 红球
data = []

while len(data) != 6:
    x = random.randint(1, 33)
    if x not in data:
        data.append(x)
    else:
        continue
print("红球：", data)

# 蓝球
x = random.randint(1, 16)
print("蓝球：", x)