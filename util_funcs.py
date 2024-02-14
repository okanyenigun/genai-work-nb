import pickle
import pandas as pd


def indexing_1(source_data, new_data, indexed_title, map_title, new_column):
    source = source_data.copy()
    try:
        source.set_index(indexed_title, inplace=True)
    except:
        pass
    map_dict = source[map_title].to_dict()
    new_data[new_column] = new_data[indexed_title].map(map_dict)
    return new_data


def read_pickle_file(path):
    infile = open(path, "rb")
    data = pickle.load(infile)
    infile.close()
    return data

def match_zero_label_w_df(pickle_file, excel_file, labels):
    data = read_pickle_file(pickle_file)
    temp = pd.DataFrame(columns=["deep_translator"] + labels)
    for row in data:
        dicto = list(row.values())[0]
        if not dicto:
            continue
        line = [dicto["sequence"]]

        for label in labels:
            idx = dicto["labels"].index(label)

            line.append(dicto["scores"][idx])
        temp.loc[len(temp), :] = line
    df = pd.read_excel(excel_file)
    for label in labels:
        df = indexing_1(temp, df, "deep_translator", label, "label_"+label)
    return df