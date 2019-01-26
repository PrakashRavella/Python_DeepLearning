s = input("Input a string")
d=l=w=0
for c in s:
    if c.isdigit():
        d=d+1

    elif c.isalpha():
        l=l+1


print("Words", len(s.split()))
print("Letters", l)
print("Digits", d)
