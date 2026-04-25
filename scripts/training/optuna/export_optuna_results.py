import sqlite3
import json

db_path = 'data/reports/training/optuna/unit_classifier_optuna.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT study_id, study_name FROM studies")
studies = cursor.fetchall()

results = {"studies": []}

for study_id, study_name in studies:
    study_info = {"name": study_name, "trials": []}
    
    cursor.execute(f"""
        SELECT t.number, t.state, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = {study_id}
        ORDER BY t.number
    """)
    trials = cursor.fetchall()
    
    for num, state, val in trials:
        study_info["trials"].append({
            "number": num,
            "state": state,
            "value": val
        })
    
    completed = [t for t in trials if t[1] == 'COMPLETE' and t[2] is not None]
    if completed:
        best = max(completed, key=lambda x: x[2])
        study_info["best_trial"] = best[0]
        study_info["best_value"] = best[2]
    
    results["studies"].append(study_info)

conn.close()

with open("optuna_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("结果已保存到 optuna_summary.json")
