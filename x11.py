a=int(input())
b=int(input())
def is_perfect(x):
    if x<=1:
        return 0
    sum=1    
    for i in range(2, int(x**0.5)+1):
        if x%i==0:
            sum+=i
            if i != x//i:
                sum +=x//i
    return x==sum

for i in range(a,b+1):
    if is_perfect(i):
        print(i,end=" ")
