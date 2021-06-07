import numpy as np

class Parameter:

    def __init__(self, \
                 name, \
                 is_free_parameter, \
                 value = None, \
                 transform_prior_cube_coordinate = None, \
                 plot_in_corner = False, \
                 corner_ranges = None, \
                 corner_transform = None, \
                 corner_label = None):
        """
        Parameter
        This class allows easy translation between the pyMultinest hypercube and 
        the physical unit space. Each parameter includes a name, which can be used
        as a reference in the model function, a value, a flag of whether it's a free parameter,
        and if it's free, a function that translates the unit hypercube into physical space.
        The remainder of the arguments deal with the corner plots.
        """
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
        else:
            import sys
            print('Error! Parameter '+self.name+' is not a free parameter!')
            sys.exit(1)

    def set_param(self, value):

        if self.is_free_parameter:
            self.value = value
        else:
            import sys
            print('Error! Parameter '+self.name+' is not a free parameter!')
            sys.exit(1)

        
class Made_up_parameter():

    def __init__(self, \
                 name, \
                 corner_ranges = None, \
                 transform_function = None, \
                 transform_parameters = None, \
                 corner_label = None):

        self.name = name
        self.corner_ranges  = corner_ranges
        self.transform_function = transform_function
        self.transform_parameters = transform_parameters
        self.corner_label = corner_label
        
