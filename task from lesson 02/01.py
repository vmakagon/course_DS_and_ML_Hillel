Input = {'rozetka': ['iphone', 'macbook', 'ipad'], 'fua': ['macbook', 'ipad'], 'citrus': ['iphone', 'macbook', 'earpods'], 'allo': ['earpods', 'iphone']}
list_goods = []
for key in Input:
    list_goods = list_goods + Input[key]
list_goods = list(set(list_goods))
Output = dict()
for good in list_goods:
    Output[good] = []
    for key in Input:
        if good in Input[key]:
            Output[good] += [key]
print(Output)

