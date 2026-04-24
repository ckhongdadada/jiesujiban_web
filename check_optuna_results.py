import sqlite3
import os

db_path = 'data/reports/training/optuna/unit_classifier_optuna.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT study_id, study_name FROM studies")
studies = cursor.fetchall()
print(f"研究数量: {len(studies)}")

for study_id, study_name in studies:
    print(f"\n研究: {study_name}")
    
    cursor.execute(f"""
        SELECT COUNT(*) FROM trials WHERE study_id = {study_id}
    """)
    total = cursor.fetchone()[0]
    
    cursor.execute(f"""
        SELECT COUNT(*) FROM trials WHERE study_id = {study_id} AND state = 'COMPLETE'
    """)
    completed = cursor.fetchone()[0]
    
    cursor.execute(f"""
        SELECT t.number, tv.value
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = {study_id} AND t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT 5
    """)
    top_trials = cursor.fetchall()
    
    print(f"  总试验: {total}, 完成: {completed}")
    
    if top_trials:
        print(f"  Top 5 试验:")
        for i, (num, val) in enumerate(top_trials, 1):
            print(f"    {i}. Trial {num}: F1={val:.6f}")

conn.close()
