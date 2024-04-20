import math

def floor_to_ten(num):
    return int(math.floor(num / 10.0)) * 10


def ceil_to_ten(num):
    return int(math.ceil(num / 10.0)) * 10


def minimum(d, attribute): # funkcja znajdujaca minimalna wartosc podanego atrybutu
    attribute_index_in_tuple = d.columns.get_loc(attribute)
    min_value = float('inf')
    for row in d.itertuples(name=None, index=False):
        current_value = row[attribute_index_in_tuple]
        if current_value < min_value:
            min_value = current_value
    return min_value


def maximum(d, attribute): # funkcja zwracajaca maksymalna wartosc podanego atrybutu
    attribute_index_in_tuple = d.columns.get_loc(attribute)
    max_value = float('-inf')
    for row in d.itertuples(name=None, index=False):
        current_value = row[attribute_index_in_tuple]
        if current_value > max_value:
            max_value = current_value
    return max_value
