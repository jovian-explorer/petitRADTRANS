import sys
import logging
class Parameter:
    """
    Parameter
    This class allows easy translation between the pyMultinest hypercube and
    the physical unit space. Each parameter includes a name, which can be used
    as a reference in the model function, a value, a flag of whether it's a free parameter,
    and if it's free, a function that translates the unit hypercube into physical space.
    The remainder of the arguments deal with the corner plots.

    Args:
        name : string
            The name of the parameter. Must match the name used in the model function.
        is_free_parameter : bool
            True if the parameter should be sampled in the retrieval
        value : float
            The value of the parameter. Set using set_param.
        transform_prior_cube_coordinate : method
            Transform the unit interval [0,1] to the physical space of the parameter.
        plot_in_corner : bool
            True if this parameter should be included in the output corner plot
        corner_ranges : Tuple(float,float)
            The axis range of the parameter in the corner plot
        corner_transform : method
            A function to scale or transform the value of the parameter for prettier plotting.
        corner_label : string
            The axis label for the parameter, defaults to name.
    """

    def __init__(self, \
                 name, \
                 is_free_parameter, \
                 value = None, \
                 transform_prior_cube_coordinate = None, \
                 plot_in_corner = False, \
                 corner_ranges = None, \
                 corner_transform = None, \
                 corner_label = None):

        self.name = name
        self.is_free_parameter = is_free_parameter
        self.value = value
        self.transform_prior_cube_coordinate = \
            transform_prior_cube_coordinate
        self.plot_in_corner = plot_in_corner
        self.corner_ranges  = corner_ranges
        self.corner_transform = corner_transform
        self.corner_label = corner_label

    def get_param_uniform(self, cube):
        if self.is_free_parameter:
            return self.transform_prior_cube_coordinate(cube)
        logging.error('Error! Parameter '+self.name+' is not a free parameter!')
        sys.exit(1)
    def set_param(self, value):
        if self.is_free_parameter:
            self.value = value
            return
        logging.error('Error! Parameter '+self.name+' is not a free parameter!')
        sys.exit(1)