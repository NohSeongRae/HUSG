import torch
v_list=torch.rand(6,2)

print(v_list)
for i in range(len(v_list)):
    print(f"{i+1}th, {i+2}th")
    init_vertex=v_list[i]
    if i + 1 < len(v_list):
        target=vertex=v_list[i+1]
    else:
        print(v_list[i])
        print(v_list[i+1])

