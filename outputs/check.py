import json 
OLD_FILE = "F:\Research\Phenikaa_Campus_Courier_2026\AI\outputs\cac_ban_nop_cu\predictions_v32b_cv_strong_final.json"
NEW_FILE = "F:\Research\Phenikaa_Campus_Courier_2026\AI\outputs\cac_ban_nop_cu\predictions_v34d_cand.json"

ROBOTS_PER_MAP = 10 
with open(OLD_FILE, "r", encoding="utf-8") as f:
    old = json.load(f) 
with open(NEW_FILE, "r", encoding="utf-8") as f:
    new = json.load(f) 
if len(old) != len(new):
    raise ValueError("Two files have different lengths") 
changed_robots = 0
changed_maps = 0 
for map_index in range(0, len(old), ROBOTS_PER_MAP): 
    map_changes = [] 
    for robot_index in range(ROBOTS_PER_MAP): 
        i = map_index + robot_index 
        if old[i] != new[i]:
            map_changes.append(
                (robot_index + 1, old[i], new[i])
            ) 
    if map_changes:
        changed_maps += 1 
        print(f"\nMap {map_index // ROBOTS_PER_MAP + 1}") 
        for robot, old_pred, new_pred in map_changes:
            print(f"  Robot {robot}: {old_pred} -> {new_pred}") 
            changed_robots += 1 
print(f"Changed maps   : {changed_maps}")
print(f"Changed robots : {changed_robots}")
print(f"Difference     : {changed_robots / len(old) * 0.3:.4f}")