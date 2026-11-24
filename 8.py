numbers=[10,20,30,40,50,60,70]
target = 50
low=0
high=len(numbers)-1
while low<=high:
    mid=(low+high)//2
    if numbers[mid]==target:
        printf("found at index:",mid)
        breake
        elif