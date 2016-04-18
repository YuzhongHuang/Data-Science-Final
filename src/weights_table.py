column_lst = ['DMIndicator', 'Gender', 'YearOfBirth', 'Height', 'Weight',
       'SystolicBP', 'DiastolicBP', 'RespiratoryRate', 'Temperature',
       'circulatory', 'congenital', 'digestive', 'endocrine',
       'external_injury', 'genitourinary', 'infectious',
       'injury_poisoning', 'mental_disorders', 'musculoskeletal',
       'neoplasms', 'nervous', 'perinatal', 'pregnancy', 'respiratory',
       'sense', 'skin', 'symptoms']

# df = pandas.DataFrame(index=range(da.n_hidden), columns=column_lst)

# for i in range(len(column_lst)):
#     df[column_lst[i]] = weights[i]

# df.to_csv("../data/processed/weights_table.csv")