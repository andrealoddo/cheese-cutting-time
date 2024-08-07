def get_dict_from_classreportstringdict(d):
    d = d.replace('{', '')
    d = d.replace('}', '')
    d = d.replace('\'', '')
    result = dict((a, float(b.strip()))
                  for a, b in (element.split(': ')
                               for element in d.split(', ')))
    return result
