from pdk_learning import *
from sklearn.model_selection import train_test_split
import MDAnalysis as mda
import numpy as np


class ProteinDataSet:
    """
    Class representing protein data.

    Attributes
    -----------
    trajectory_filename : str
        Path to trajectory file.
    topology_filename : str
        Path to topology file.
    config_parameters : str, optional
        Protein's configuration parameters. Default is None.
    """

    def __init__(
        self,
        trajectory_filename,
        topology_filename,
        target_property_filename,
        config_parameters=None,
    ):
        """
        Initialise ProteinData object.

        Parameters
        -----------
        trajectory_filename : str
            Path to trajectory file.
        topology_filename : str
            Path to topology file.
        config_parameters : str, optional
            Protein's configuration parameters. Default is None.
        """

        self.trajectory_filename = trajectory_filename
        self.topology_filename = topology_filename
        self.target_property_filename = target_property_filename
        self.trajectory_data = self._read_trajectory(
            self.trajectory_filename, self.topology_filename
        )
        self.topology_data = self._read_topology(self.topology_filename)
        self.ca_atom_group = self._select_CA_atoms()
        self.n_frames = self.trajectory_data.trajectory.n_frames
        self.frames = self._frames_of_trajectory()
        self.frame_indices = self._frame_indices_of_trajectory()
        self.ref_coordinates = self.topology_data.trajectory[0].positions
        self.target_property = self.read_target_property(self.target_property_filename)

    def _read_topology(self, topology_filename):
        """
        Load topology file.

        Parameters
        ----------
        topology_filename : str
            Path to topology file.

        Returns
        -------
        mda.Universe
            An instance of the MDAnalysis Universe representing the loaded topology data.
        """
        top_data = mda.Universe(topology_filename)
        return top_data

    def _read_trajectory(self, trajectory_filename, topology_filename):
        """
        Load trajectory and topology files into Universe to build the object.

        Parameters
        -----------
        trajectory_filename : str
            Path to trajectory file.
        topology_filename : str
            Path to topology file.

        Returns
        -----------
        mda.Universe
            An instance of the MDAnalysis Universe representing the loaded trajectory.
        """
        trajectory_data = mda.Universe(
            topology_filename,
            trajectory_filename,
            permissive=False,
        )
        return trajectory_data

    def _select_CA_atoms(self):
        """
        Select C-alpha atoms from the first frame of the trajectory.

        Returns
        -------
        MDAnalysis.core.groups.AtomGroup
            An AtomGroup containing the C-alpha atoms from the first frame of the trajectory.
        """
        ca_atom_group = self.trajectory_data.select_atoms("name CA")
        return ca_atom_group

    def _frames_of_trajectory(self):
        """
        Generate a dictionary with frame numbers and timesteps for a protein trajectory.

        Returns
        -------
        list of tuples
            A list of tuples containing the frame number (index) and corresponding timestep for each frame.
        """
        frames = []
        for x in range(len(self.trajectory_data.trajectory)):
            _ = self.trajectory_data.trajectory[x]
            frames.append(
                (
                    x,
                    self.trajectory_data.trajectory.ts.from_timestep(
                        self.trajectory_data.trajectory[x]
                    ),
                ),
            )

        return frames

    def _frame_indices_of_trajectory(self):
        """
        Generate a list of frame indices for a protein trajectory.

        Returns
        -------
        list
            A list of integers representing the frame indices from the protein trajectory.

        """
        frame_indices = []
        for x in range(len(self.trajectory_data.trajectory)):
            frame_indices.append(x)
        return frame_indices

    def frame_selection_iterator(self, selection_of_frames):
        """
        Create a new object with similar attributes to a trajectory object from a specific selection of frames.

        Parameters
        ----------
        selection_of_frames : int or slice
            Single frame or slice of frames from the trajectory to select.

        Returns
        -------
        FrameIteratorIndices
            An instance of the MDAnalysis.coordinates.base.FrameIteratorIndices.
            It is iterable over the frames of a trajectory.

        Raises
        ------
        TypeError
            If the `selection_of_frames` parameter is neither an integer nor a slice.

        Notes
        -----
        The method creates a boolean mask array to indicate the selected frames.
        If an integer or slice is provided, the corresponding indices in the mask are set to True.
        The selected frames are extracted from the trajectory data using the mask.
        """
        trajectory_data = self.trajectory_data.trajectory
        mask = np.array([False for _ in trajectory_data])
        for i in selection_of_frames:
            if isinstance(i, int) or isinstance(i, slice):
                mask[i] = True
            else:
                raise TypeError("Expected int or slice")
        selected_frames = trajectory_data[np.where(mask)[0]]
        return selected_frames

    def frame_selection_indices(self, selection_of_frames):
        """
        Generate a list with only selected frames from a protein trajectory

        Parameters
        -----------
        selection_of_frames : int or slice
            Single frame or slice of frames from the trajectory to select.

        Returns
        -------
        List
            Contains indices of selected frames.

        Raises
        ------
        TypeError
            If the `selection_of_frames` parameter is neither an integer nor a slice.
        """
        trajectory_data = self.trajectory_data.trajectory
        mask = np.array([False for _ in trajectory_data])
        for i in selection_of_frames:
            if isinstance(i, int) or isinstance(i, slice):
                mask[i] = True
            else:
                raise TypeError("Expected int or slice")
        selected_frames = trajectory_data[np.where(mask)[0]]
        indices_of_selected_frames = [ts.frame for ts in selected_frames]
        return indices_of_selected_frames

    def write_xtc_file(self, outfilepath, selected_frames):
        """
        Generate an xtc file containing only selected frames from a protein trajectory.

        Parameters
        -----------
        outfilepath : str
            Path where output file is saved.
        selected_frames : int ot list,
            Single frame or list of frames from trajectory.
        """
        protein = self.trajectory_data.select_atoms("protein")
        with mda.Writer(outfilepath, protein.n_atoms) as W:
            for t_idx in selected_frames:
                self.trajectory_data.trajectory[t_idx]
                W.write(protein)

    def cast_output_traj_to_numpy(self, outfilepath, subsampled_traj, unit="nanometer"):
        """
        Casts an XTC file into a NumPy array for user readability.

        Parameters
        -----------
        outfilepath : str
            Path where output file is saved.
        subsampled_traj : MDAnalysis.coordinates.XTC.XTCReader
            XTC trajectory file.
        unit : str, optional
            Unit for coordinates valuess.

        Returns
        -------
        numpy.ndarray
            NumPy array containing the coordinates of the subsampled trajectory.

        """
        coordinates_numpy = []
        for ts in subsampled_traj:
            coordinates_numpy.append(deepcopy(ts.positions))
        coordinates_numpy = np.array(coordinates_numpy)
        if unit == "nanometer":
            coordinates_numpy = coordinates_numpy / 10
        np.save(outfilepath, coordinates_numpy)
        return coordinates_numpy

    def convert_numpy_to_2D(self, infilepath, outfilepath):
        """
        Convert a 3D numpy array to a 2D numpy array and save it to a file.

        Parameters
        ----------
        infilepath : numpy.ndarray
            The input 3D numpy array to be converted.
        outfilepath : str
            The path where the output file will be saved.

        Returns
        -------
        numpy.ndarray
            The converted 2D numpy array.
        """
        (x, y, z) = infilepath.shape
        outfile = np.reshape(infilepath, (x, y * z))
        np.save(outfilepath, outfile)
        return outfile

    def _get_holdout_indices(self, test_set_size, validation_set_size=0.0):
        """
        Generate indices for trainin, testing and validation rows

        Parameters
        ----------
        test_set_size : float, opt
            Path to the input file containing the data to be split.
        """
        train_val_indices, test_indices = train_test_split(
            self.frame_indices, test_size=test_set_size, random_state=25
        )

        if validation_set_size > 0.0:
            # Adjust validation set size to be relative to the (training + validation) size
            adjusted_validation_size = validation_set_size / (1.0 - test_set_size)

            # Split the remaining data into training and validation sets
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=adjusted_validation_size, random_state=25
            )
        else:
            train_indices = train_val_indices
            val_indices = []

        return train_indices, val_indices, test_indices

    def create_holdout_data_set(self, test_set_size=0.3, validation_set_size=0.0):
        """
        split dataset into training, testing and validation sets

        Parameters
        ----------
        infilepath : numpy.ndarray
            The input 3D numpy array to be converted.
        outfilepath : str
            The path where the output file will be saved.

        Returns
        -------
        numpy.ndarray
            The converted 2D numpy array.
        """
        if test_set_size < 0.0 or test_set_size > 0.5:
            raise ValueError("test size should be in the range [0, 0.5]")

        if validation_set_size < 0.0 or validation_set_size > (1.0 - test_set_size):
            raise ValueError(
                f"validation size should be in the range [0, {1.0 - test_set_size}]"
            )

        training_indices, validation_indices, test_indices = self._get_holdout_indices(
            test_set_size, validation_set_size
        )

        ml_data_set = MLDataSet(self)
        ml_data_set.training_indices = training_indices
        ml_data_set.validation_indices = validation_indices
        ml_data_set.test_indices = test_indices

        return ml_data_set

    def read_target_property(self, target_property_filename):
        target_property = np.loadtxt(target_property_filename)
        return target_property

    def get_indices_target(self, target_property_filename):
        frame_indices = []
        for x in range(len(target_property_filename)):
            frame_indices.append(x)
        return frame_indices

    def filter_target_indices(self, selection_of_frames):
        selection_of_frames = self.create_holdout_data_set().training_indices
        mask = np.array([False for _ in selection_of_frames])
        for i in selection_of_frames:
            if isinstance(i, int) or isinstance(i, slice):
                mask[i] = True
            else:
                raise TypeError("Expected int or slice")
        selected_frames = selection_of_frames[np.where(mask)[0]]
        filtered_target_indices = [ts.frame for ts in selected_frames]
        return filtered_target_indices
