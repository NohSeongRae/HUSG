import torch
import numpy as np
v_list=torch.rand(6,2)

normals=[]
print(v_list)
for i in range(len(v_list)):
    print(f"{i+1}th, {i+2}th")
    init_vertex=v_list[i]
    target_vertex=v_list[(i+1)%len(v_list)]
    edge=target_vertex-init_vertex
    normal=torch.tensor([-edge[1], edge[0]])
    unit_normal = normal / np.linalg.norm(normal)
    normals.append(unit_normal)

print(normals)