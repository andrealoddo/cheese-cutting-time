from utils.folders import make_path, create_folder


# Create an object named FolderHandler and inherit from the object class
class FolderHandler(object):
    root = ""
    data = ""
    results = ""
    results_std = ""
    results_pickle = ""
    latex = ""
    plots = ""

    def __init__(self, features_root, exp_folder, exp, set_folder):
        # create a variable named path and set it to path
        self.root = make_path('..', '..', 'Experiments', exp_folder, exp, set_folder)
        self.data = features_root
        self.results = make_path(self.root, 'Results')
        self.results_std = make_path(self.root, 'ResultsStd')
        self.results_pickle = make_path(self.root, 'ResultsPickle')
        self.plots = make_path(self.root, 'Plots')
        self.latex = make_path(self.root, 'Latex')

        create_folder(self.results)
        create_folder(self.results_std)
        create_folder(self.results_pickle)
        create_folder(self.plots)
        create_folder(self.latex)



