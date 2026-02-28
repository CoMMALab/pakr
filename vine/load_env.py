import numpy as np
def load_box_config(filename: str):
    """
    Reads a config file with fields:
      bound: <float>
      start: <float> <float> <float>
      goal:  <float> <float> <float>
      ob_type: box
      obstacles:
        x1 y1 x2 y2
        ...
    Returns a dict with bound, start, goal, ob_type, obstacles (Nx4).
    """

    cfg = {
        'bound_x': None,
        'bound_y': None,
        'start': None,
        'goal': None,
        'scale': 1.0,
        'ob_type': None,
        'obstacles': []
    }
    reading_obstacles = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if reading_obstacles:
                parts = line.split()
                if len(parts) == 4:
                    cfg['obstacles'].append([float(p) for p in parts])
                continue

            if line.startswith("bound:"):
                cfg['bound_x'] = float(line.split(' ')[1].strip())
                cfg['bound_y'] = float(line.split(' ')[2].strip())
            elif line.startswith("start:"):
                values = line.split(':')[1].split()
                cfg['start'] = [float(v) for v in values]
            elif line.startswith("goal:"):
                values = line.split(':')[1].split()
                cfg['goal'] = [float(v) for v in values]
            elif line.startswith("goal_radius:"):
                cfg['goal_radius'] = float(line.split(':')[1].strip())
            elif line.startswith("ob_type:"):
                cfg['ob_type'] = line.split(':')[1].strip()
            elif line.startswith("scale:"):
                cfg['scale'] = float(line.split(':')[1].strip())
            elif line.startswith("obstacles:"):
                reading_obstacles = True


    # Make sure for obstacles, x1 < x2 and y1 < y2
    for i in range(len(cfg['obstacles'])):
        x1, y1, x2, y2 = cfg['obstacles'][i]
        if x1 > x2:
            cfg['obstacles'][i][0] = x2
            cfg['obstacles'][i][2] = x1
        if y1 > y2:
            cfg['obstacles'][i][1] = y2
            cfg['obstacles'][i][3] = y1
    
    cfg['obstacles'] = np.array(cfg['obstacles'], dtype=np.float32)
    
    # If lack goal radius, set to 100
    if 'goal_radius' not in cfg:
        cfg['goal_radius'] = 100.0 / cfg['scale']

    # Scale the obstacles
    cfg['obstacles'][:, :] *= cfg['scale']
    cfg['start'] = [cfg['start'][0] * cfg['scale'], cfg['start'][1] * cfg['scale'], cfg['start'][2]]
    cfg['goal'] = [cfg['goal'][0] * cfg['scale'], cfg['goal'][1] * cfg['scale'], cfg['goal'][2]]
    cfg['goal_radius'] = cfg['goal_radius'] * cfg['scale']
    cfg['bound_x'] *= cfg['scale']
    cfg['bound_y'] *= cfg['scale']
    
    return cfg