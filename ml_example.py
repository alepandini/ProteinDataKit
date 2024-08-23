from pdk_protein import ProteinDataSet

trajectory_file = "data/MD_4AKEA_protein.xtc"
topology_file = "data/MD_4AKEA_protein.pdb"
target_property_file = "data/MD_4AKEA_protein.dat"

p_data = ProteinDataSet(
    trajectory_file,
    topology_file,
    target_property_file,
    config_parameters=None,
)

target_property = p_data.target_property
target_indices = p_data.get_indices_target(target_property)
# print(target_indices)
ml_obj = p_data.create_holdout_data_set()
selection = ml_obj.training_indices
print(len(selection))
filter_target = p_data.filter_target_indices(selection)

print(len(filter_target))
