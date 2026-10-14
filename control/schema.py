import pandas as pd
import pandera.pandas as pa




schema = pa.DataFrameSchema({
"kund_id": pa.Column(int, [pa.Check.ge(1)], unique=True), 

"alder": pa.Column(int, [pa.Check.ge(1), pa.Check.le(100)]), 

"region": pa.Column(str, [pa.Check.isin(["Mitt", "Syd", "Väst"])]), 

"abonemang": pa.Column(str, [pa.Check.isin(["Bas", "Standard", "Premium"])]),

"mander_som_kund": pa.Column(float, [pa.Check.ge(0)]),

"manadsavgift": pa.Column(float, [pa.Check.ge(0)]), 

"data_gb_per_manad": pa.Column(int, [pa.Check.ge(0)]), 

"supportarenden_12m": pa.Column(int, [pa.Check.ge(0)]), 

"avslutat": pa.Column(int, [pa.Check.isin([0, 1])]), 


})

df = pd.read_csv("data/kundbas_ny_leverans.csv")

try:
    validate_df = schema.validate(df, lazy=True)
    print("validering ok")
    print(validate_df)
except pa.errors.SchemaErrors as e:
    print("validering ej ok")
    print(e.failure_cases)