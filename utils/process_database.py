from pathlib import Path
import numpy as np
import sqlite3
import shutil

from .database import pair_id_to_image_ids, blob_to_array
from .database import COLMAPDatabase


def read_two_view_from_db(database_path):
    """Read two-view geometry data from COLMAP database."""
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    pairs, matches, F_list, E_list, H_list, pairs_id = [], [], [], [], [], []

    query = """SELECT pair_id, rows, cols, data, F, E, H, qvec, tvec FROM two_view_geometries"""
    for row in db.execute(query):
        pair_id, rows, cols, data, F, E, H, qvec, tvec = row
        if data is None:
            continue
        id1, id2 = pair_id_to_image_ids(pair_id)
        pairs.append((id2name[id1], id2name[id2]))
        matches.append(blob_to_array(data, np.uint32, (rows, cols)))
        F_list.append(blob_to_array(F, np.float64).reshape(-1, 3))
        E_list.append(blob_to_array(E, np.float64).reshape(-1, 3))
        H_list.append(blob_to_array(H, np.float64).reshape(-1, 3))
        pairs_id.append(pair_id)

    db.close()
    return pairs, matches, E_list, F_list, H_list, pairs_id

def delete_multiple_records(database_path, match_pair_list, table="two_view_geometries"):
    """Delete multiple records from the specified database table."""
    try:
        db = COLMAPDatabase.connect(database_path)
        cursor = db.cursor()
        cursor.executemany(f"DELETE FROM {table} WHERE pair_id = ?", [(i,) for i in match_pair_list])
        db.commit()
        print(f"Deleted {cursor.rowcount} records from {table}.")
        cursor.close()
    except sqlite3.Error as e:
        print(f"Error deleting records: {e}")
    finally:
        if db:
            db.close()


def remove_doppelgangers(db_path, pair_probability_file, pair_path, threshold):
    """Filter pairs based on probability threshold."""
    new_db_path = db_path.replace('.db', f'_threshold_{threshold:.3f}.db')
    shutil.copyfile(db_path, new_db_path)

    results = np.load(pair_probability_file, allow_pickle=True).item()
    y_scores = np.array(results['prob'])
    pairs_info = np.load(pair_path)
    pairs_id = np.array(pairs_info)[:, -1]

    print('number of matches in database: ', len(y_scores))
    match_pair_list = [pairs_id[i] for i in range(len(pairs_info)) if y_scores[i] < threshold]
    for table in ["matches", "two_view_geometries"]:
        delete_multiple_records(new_db_path, match_pair_list, table=table)

    return new_db_path


def create_image_pair_list(db_path, output_path):
    pairs_list = []
    pairs, matches, _, _, _, pairs_id = read_two_view_from_db(db_path)
    for i in range(len(pairs_id)):
        name1, name2 = pairs[i]
        label = 0
        pairs_list.append([name1, name2, label, matches[i].shape[0], pairs_id[i]])
    pairs_list = np.concatenate(pairs_list, axis=0).reshape(-1, 5)
    np.save('%s/pairs_list.npy' % output_path, pairs_list)    
    return '%s/pairs_list.npy' % output_path



