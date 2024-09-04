<h1 align='center'>Dynamic PDB: A New Dataset and a SE(3) Model Extension by Integrating Dynamic Behaviors and Physical Properties in Protein Structures</h1>

<div align='center'>
    Ce Liu<sup>1*</sup>&emsp;
    Jun Wang<sup>1*</sup>&emsp;
    Zhiqiang Cai<sup>1*</sup>&emsp;
    Yingxu Wang<sup>1,3</sup>&emsp;
    Huizhen Kuang<sup>2</sup>&emsp;
    Kaihui Cheng<sup>2</sup>&emsp;
    Liwei Zhang<sup>1</sup>&emsp;
</div>
<div align='center'>
    Qingkun Su<sup>1</sup>&emsp;
    Yining Tang<sup>2</sup>&emsp;
    Fenglei Cao<sup>1</sup>&emsp;
    Limei Han<sup>2</sup>&emsp;
    <a href='https://sites.google.com/site/zhusiyucs/home/' target='_blank'>Siyu Zhu</a><sup>2†</sup>&emsp;
    Yuan Qi<sup>2†</sup>&emsp;
</div>

<div align='center'>
    <sup>1</sup>Shanghai Academy of Artificial Intelligence for Science&emsp;
    <sup>2</sup>Fudan University&emsp;
    <br>
    <sup>3</sup>Mohamed bin Zayed University of Artificial Intelligence
</div>

<table>
  <tr>
    <td align="center">
      <img src="assets/3tvj.png" alt="protein example" style="width: 80%;">
    </td>
  </tr>
</table>

## Overview

**Dynamic PDB** is a large-scale dataset that enhances existing prestigious static 3D protein structural databases, such as the Protein Data Bank (PDB), by integrating dynamic data and additional physical properties. It contains approximately 12.6k filtered proteins, each subjected to all-atom molecular dynamics (MD) simulations to capture conformational changes.

Compared with previously existing protein MD datasets, dynamic PDB provides three key advancements:

1. **`Extended simulation durations`**: Up to 1 microsecond per protein, facilitating a more comprehensive understanding of significant conformational changes.
2. **`Finer-grained sampling intervals`**: 1 picosecond intervals, allowing for the capture of more detailed allosteric pathways.
3. **`Enriched array of physical properties`**: Captured during the MD process, including atomic velocities and forces, potential/kinetic energies, and the temperature of the simulation environment, etc.

## What dynamic PDB contains?

The attributes contained in dynamic PDB are listed as follows:

| File Name | Attribute | Data Type | Unit |
| --- | --- | --- | --- |
| `{protein_id}_T.pkl` | Trajectory coordinates | float array | Å |
| `{protein_id}_V.pkl` | Atomic velocities | float array | Å/ps |
| `{protein_id}_F.pkl` | Atomic forces | float array | kcal/mol·Å |
| `{protein_id}_npt_sim.dat` | Potential energy<br>Kinetic energy<br>Total energy<br>Temperature<br>Box volume<br>System density | float<br>float<br>float<br>float<br>float<br>float | kJ/mole<br>kJ/mole<br>kJ/mole<br>K<br>nm³<br>g/mL |

In addition, the following data are stored during the MD simulation:

| File Name | Description |
| --- | --- |
| `{protein_id}_minimized.pdb` | PDB structure after minimization |
| `{protein_id}_nvt_equi.dat` | Information in NVT equilibration |
| `{protein_id}_npt_equi.dat` | Information in NPT equilibration |
| `{protein_id}_T.dcd` | DCD format for trajectory coordinates |
| `{protein_id}_state_npt1000000.0.xml` | Status file for MD prolongation |

## Download dataset

You can easily get dynamic PDB dataset from our [ModelScope repo](https://www.modelscope.cn/datasets/fudan-generative-vision/dynamicPDB/).

Clone the dataset into `${DATA_ROOT}/dynamicPDB` directory by cmd below:
```shell
git lfs install
git clone git clone https://www.modelscope.cn/datasets/fudan-generative-vision/dynamicPDB.git dynamicPDB
```

Finally, the dataset should be organized as follows:

```text
./dynamicPDB/
|-- 1ab1_A_npt1000000.0_ts0.001
|   |-- 1ab1_A_npt_sim_data
|   |   |-- 1ab1_A_npt_sim_0.dat
|   |   `-- ...
|   |-- 1ab1_A_dcd
|   |   |-- 1ab1_A_dcd_0.dcd
|   |   `-- ...
|   |-- 1ab1_A_T
|   |   |-- 1ab1_A_T_0.pkl
|   |   `-- ...
|   |-- 1ab1_A_F
|   |   |-- 1ab1_A_F_0.pkl
|   |   `-- ...
|   |-- 1ab1_A_V
|   |   |-- 1ab1_A_V_0.pkl
|   |   `-- ...
|   |-- 1ab1_A.pdb
|   |-- 1ab1_A_minimized.pdb
|   |-- 1ab1_A_nvt_equi.dat
|   |-- 1ab1_A_npt_equi.dat
|   |-- 1ab1_A_T.dcd
|   |-- 1ab1_A_T.pkl
|   |-- 1ab1_A_F.pkl
|   |-- 1ab1_A_V.pkl
|   `-- 1ab1_A_state_npt1000000.0.xml
|-- 1uoy_A_npt1000000.0_ts0.001
|   |-- ...
|   `-- ...
`-- ...
```

## Applications