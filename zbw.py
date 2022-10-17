num_list = [1, 2, 3, 4]

pos = 1234

_filter = [True * 4322]
prime_number = []
num = 2
print(_filter)
while num < 4321:
    if _filter[num]:
        prime_number.append(num)
    pos = 0
    while pos < len(prime_number) and num * prime_number[pos] < 4321 and num % prime_number[pos] == 0:
        _filter[num * prime_number[pos]] = False
    num += 1

while pos < 4321:
    # 本质是个map
    pre_list = [int(str(pos)[0]), int(str(pos)[1]), int(str(pos)[2]), int(str(pos)[3])].sort()
    if pre_list == num_list:
        if prime_number[pos]:
            print(pos)
