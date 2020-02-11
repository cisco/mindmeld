from mindmeld.converter.dialogflow import DialogflowConverter

df = '/Users/vijay/mindmeld/tests/converter/dialogflow_sample_project'
trg = '/Users/vijay/mm_banking'

df_init = DialogflowConverter(df, trg)
df_init.convert_project()
