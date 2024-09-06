# deal with periodic

import openmm
from openmm import app
from openmm import unit

import mdtraj as md
import numpy as np

def get_ckpt_boxsize(pdb_path,ckpt_path):
    pdb = app.PDBFile(pdb_path)
    
    forcefield_name = 'amber14/protein.ff14SB.xml'
    solvent_name = 'amber14/tip3p.xml'
    forcefield = app.ForceField(forcefield_name, solvent_name)
    # add hydrogen
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0)
    topology, positions = modeller.getTopology(), modeller.getPositions()
    # add solvent (see http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html?highlight=padding)
    box_padding = 1.0 * unit.nanometers
    ionicStrength = 150 * unit.millimolar # atlas NaCl density
    positiveIon = 'Na+' 
    negativeIon = 'Cl-'
    modeller = app.Modeller(topology, positions)
    modeller.addSolvent(forcefield, 
                        model='tip3p', # atlas density
                        # boxSize=openmm.Vec3(5.0, 5.0, 5.0) * unit.nanometers,
                        padding=box_padding,
                        ionicStrength=ionicStrength,
                        positiveIon=positiveIon,
                        negativeIon=negativeIon,
        )
    topology, positions = modeller.getTopology(), modeller.getPositions()
    
    system = forcefield.createSystem(topology, nonbondedMethod=app.PME, constraints=None,
                                    rigidWater=None)

    integrator = openmm.VerletIntegrator(0.001) # middle leap-frog
    # integrator = openmm.VerletIntegrator(timestep)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    simulation = app.Simulation(topology, system, integrator, platform) 
    
    if ckpt_path:
        simulation.loadCheckpoint(ckpt_path)
    else:
        print('with out ckpt we may get error peridic box size')
        
    return simulation.system.getDefaultPeriodicBoxVectors()

def rebuild_traj_data(traj_path,pdb_path,box_size):
    movement_threhold = abs(box_size / 2)
    
    traj = md.load(traj_path,top=pdb_path)
    traj_coords = traj.xyz
    
    movement = traj_coords[1:] - traj_coords[:-1]
    
    whole_movement = movement.mean(axis=-2) # frame-1, 3
    jump_index = np.where(abs(whole_movement) > movement_threhold)
    
    # 注明此时的蛋白质位于那个溶液盒子中，初始盒子(0,0,0)
    box_index = np.zeros((traj_coords.shape[0],3))
    
    for i,d in zip(*jump_index):
        if whole_movement[i,d] < 0:
            box_index[i+1:,d] += 1
        if whole_movement[i,d] > 0:
            box_index[i+1:,d] -= 1
    
    xyz = traj.xyz
    xyz = xyz + (box_index * box_size)[:,None,:]
    
    new_traj = md.Trajectory(xyz,topology=traj.topology)
    
    return new_traj
    
if __name__ == '__main__':
    use_gpu = True 
    
    traj_path = './2erl_A_T.dcd'
    pdb_path = './2erl_A.pdb'
    
    ckpt_path = './2erl_A_checkpoint.chk'
    
    # 没有checkpoint的输入none，结果在极端情况下不准确，不过我还没遇到过
    x,y,z = get_ckpt_boxsize(pdb_path,ckpt_path)
    assert x._value[0] == y._value[1] == z._value[2]
    box_size = abs(x._value[0])
    
    # 新输出的路径
    new_traj = rebuild_traj_data(traj_path,pdb_path,box_size)
    save_path = './2erl_A_post.dcd'
    new_traj.save_dcd(save_path)
    
    
    # 检查，此时应该所有的坐标差不超过半个盒子大小
    xyz = new_traj.xyz
    diff = xyz[1:] - xyz[:-1]
    print(abs(diff).max())
    
    print('test over')
