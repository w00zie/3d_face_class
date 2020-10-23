import numpy as np
import torch
import trimesh
import os

source_dir = "bosphorusReg3DMM/"
export_dir = "bosphorus_mesh/"
os.mkdir(export_dir)
faces = np.load("tri.npy")
files = [f for f in os.listdir(source_dir) if f.endswith('.pt')]
for f in files:
    name = f.split(".")[0]
    vertices = torch.load(source_dir+f).numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(export_dir+"{}.obj".format(name))
