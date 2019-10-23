# 3. Дан словарь продавцов и цен на iphone xs max 256gb у разных продавцов на hotline:
# { ‘citrus’: 47.999, ‘istudio’ 42.999,
#  ‘moyo’: 49.999, ‘royal-service’: 37.245,
# ‘buy.ua’: 38.324, ‘g-store’: 37.166,
# ‘ipartner’: 38.988, ‘sota’: 37.720 }
# Средствами python, написать функцию, возвращающую список имен продавцов,
# чьи цены попадают в диапазон (from_price, to_price). Например:
#
# (37.000, 38.000) -> [‘royal-service’, ‘g-store’, ‘sota’]
d = { 'citrus': 47.999, 'istudio': 42.999,
'moyo': 49.999, 'royal - service': 37.245,
'buy.ua': 38.324, 'g - store': 37.166,
'ipartner': 38.988, 'sota': 37.720}
print(d)
a = []
from_price = int(input())
to_price = int(input())
for saller,price in d.items():
    if price >= from_price and price <= to_price:
        a.append(saller)
print(a)

