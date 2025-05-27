
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter


pdb_file = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/protein/workspace/chengkaihui/code/DFOLDv2/results/val_outputs/align_to_ref_frame_11/02D_06M_2024Y_22h_51m_05s/step_100/sample/6j56_A_aligned.pdb'

# 加载包含多个模型的PDB文件
u = mda.Universe(pdb_file)

# 提取第一个模型作为参考结构
reference = u.select_atoms("protein")
reference.write("reference.pdb")

# 初始化XTC Writer
with XTCWriter("trajectory.xtc", n_atoms=reference.n_atoms) as writer:
    for ts in u.trajectory:
        writer.write(reference)

print("Reference PDB and trajectory XTC successfully written.")