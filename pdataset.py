from sklearn.model_selection import train_test_split
import MDAnalysis as mda
import numpy as np
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
        self.trajectory_data = self._read_trajectory(
            self.trajectory_filename, self.topology_filename
        )
        self.topology_data = self._read_topology(self.topology_filename)
        self.ca_atom_group = self._select_CA_atoms()
        self.n_frames = self.trajectory_data.trajectory.n_frames
        self.frames = self._frames_of_trajectory()
        self.frame_indices = self._frame_indices_of_trajectory()
        self.ref_coordinates = self.topology_data.trajectory[0].positions

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
            topology_format="GRO",
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
                log.error(
                    "{:15s} Expected int or slice in frame_selection_iterator()".format(
                        "STEPS"
                    )
                )
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
                log.error(
                    "{:15s} Expected int or slice in frame_selection_iterator()".format(
                        "STEPS"
                    )
                )
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

    def holdout_input_prep(self, infilepath, outfilepath_training, outfilepath_testing):
        """
        Prepares input data for machine learning by splitting the input file into training and testing data.

        Parameters
        ----------
        infilepath : str
            Path to the input file containing the data to be split.
        outfilepath_training : str
            Path where the training data file will be saved.
        outfilepath_testing : str
            Path where the testing data file will be saved.
        """
        training_data, testing_data = train_test_split(
            infilepath, test_size=0.3, random_state=25
        )
        np.save(outfilepath_training, training_data)
        np.save(outfilepath_testing, testing_data)
