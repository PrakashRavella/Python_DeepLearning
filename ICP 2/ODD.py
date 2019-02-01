items = []
for i in range(100, 500):
    s = str(i)
    if (int(s[0])%2==1) and (int(s[1])%2==1) and (int(s[2])%2==1):
        items.append(s)
print( ",".join(items))