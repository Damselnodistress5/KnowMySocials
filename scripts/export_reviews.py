import pandas as pd
from src.utils import get_db_engine
from sqlalchemy import text
import config

engine = get_db_engine()
sql = f'''
SELECT pr.*
FROM {config.PROCESSED_DATA_TABLE} pr
ORDER BY pr.id
'''
with engine.connect() as conn:
    res = conn.execute(text(sql))
    rows = res.fetchall()
    cols = list(res.keys())

if rows:
    df = pd.DataFrame(rows, columns=cols)
    out = 'data/reviews_after_clustering.csv'
    df.to_csv(out, index=False)
    print('Wrote', out)
    print('\nFirst 10 rows:')
    print(df.head(10).to_string(index=False))
else:
    print('No rows returned')
