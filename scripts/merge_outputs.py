import json
import os
import pandas as pd

def load_json(path):
    # load variable from json
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data

main_folder = 'intermediate'
type_label_dict_osse = {
    'time_24h': ["optimum", "gprm", "gprm_e", "gos"],
    'time_1h': ["optimum", "gprm", "gprm_e", "gos"],
    'noise': ["gprm", "gprm_e", "gos"],
    'cloud_dense': ["gprm", "gprm_e"],
    'cloud_sparse': ["gprm", "gprm_e"],
    'satellite': ["gprm_e"]
}

for type, labels in type_label_dict_osse.items():
    print(f"Processing {type}...")
    
    for label in labels:
        if type == 'satellite':
            file_names = [
                f for f in os.listdir(main_folder)
                if f.endswith('.json') and "satellite" in f
            ]
        else:
            if label == 'gprm':
                file_names = [
                    f for f in os.listdir(main_folder)
                    if f.endswith('.json') and label in f and "gprm_e" not in f
                ]
            else:    
                file_names = [
                    f for f in os.listdir(main_folder)
                    if f.endswith('.json') and label in f
                ]

        list_data = []

        for file_name in file_names:
            
            nested_dict = load_json(f'{main_folder}/{file_name}')
            if label == 'gprm_e' or label == 'gprm' or label == 'optimum' or label == "GPRM_e":
                flat_dict = {**nested_dict, **nested_dict['est_params']}
                del flat_dict['est_params']
            else:
                flat_dict = nested_dict
                
            if type == 'noise':
                step_name = "noise_sd"
                flat_dict[step_name] = flat_dict.pop("step")
            elif type == 'cloud_dense':
                step_name = "coverage_dense"
                flat_dict[step_name] = flat_dict.pop("step")
            elif type == 'cloud_sparse':
                step_name = "coverage_sparse"
                flat_dict[step_name] = flat_dict.pop("step")
            elif type == "time_1h" or type == "time_24h":
                step_name = "time_sec"
                flat_dict[step_name] = flat_dict.pop("step")
            elif type == "satellite":
                step_name = "time"
                flat_dict[step_name] = flat_dict.pop("datet")
                
                date_str = flat_dict[step_name]
                date_part, time_part = date_str.split('_')
                time_part_fixed = time_part.replace('-', ':')
                flat_dict[step_name] = f"{date_part}T{time_part_fixed}"
            list_data.append(flat_dict)
                
        if type == "time_1h" or type == "time_24h":
            rename_dict = {
                'optimum': 'gp_num_t',
                'gprm': 'gp_num_t1',
                'gprm_e': 'gp_obs_t',
                'gos': 'gos_t',
            }
        else:
            rename_dict = {
                'gprm': 'gp_num_t',
                'gprm_e': 'gp_obs_t',
                'gos': "gos_t",
                'GPRM_e': 'gp_obs_t',
            }
        if label in rename_dict:
            label_new = rename_dict[label]
        else:
            label_new = label
            
        df = pd.DataFrame(list_data)
        df = df.set_index(step_name).sort_index()
        df.to_csv(f'outputs/{type}_{label_new}.csv')
        # print(df)
