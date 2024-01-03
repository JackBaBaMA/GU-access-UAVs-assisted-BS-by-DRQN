def fib(n):
    a, b = 0, 1
    count = 1
    while count < n:
        a, b = b, a + b
        count = count + 1
        print(b,count)

c = fib(10)