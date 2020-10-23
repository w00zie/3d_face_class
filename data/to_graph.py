import os
from itertools import groupby
import trimesh
import numpy as np

src_dir = './bosphorus_mesh/'
dest_dir = '../gnn/graph_data/bosphorus_full'

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

files_src = sorted([f for f in os.listdir(src_dir) if f.endswith('obj')])
groups = {k:list(g) for k, g in groupby(files_src, key=lambda x: x[:5])}

for k, g in groups.items():
    dest_subdir = os.path.join(dest_dir, k)
    if not os.path.exists(dest_subdir):
        os.mkdir(dest_subdir)
    for elem in g:
        mesh = trimesh.load_mesh(os.path.join(src_dir, elem))
        v, e = mesh.vertices, mesh.edges
        f = mesh.faces
        v, e = np.asarray(v), np.asarray(e)
        f = np.asarray(f)
        name = elem.split('.')[0]
        np.save(os.path.join(dest_subdir, 'edges_{}.npy'.format(name)), e)
        np.save(os.path.join(dest_subdir, 'nodes_{}.npy'.format(name)), v)
        np.save(os.path.join(dest_subdir, 'faces_{}.npy'.format(name)), f)

