def indexing_1(source_data, new_data, indexed_title, map_title, new_column):
    source = source_data.copy()
    try:
        source.set_index(indexed_title, inplace=True)
    except:
        pass
    map_dict = source[map_title].to_dict()
    new_data[new_column] = new_data[indexed_title].map(map_dict)
    return new_data