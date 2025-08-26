# runAnalysis.py

import sys
from PyQt4 import QtGui

# Assuming autofp.py is available in the same directory
import autofp

def main():
    # Initialize the application
    app = QtGui.QApplication(sys.argv)

    # Check if the script is run directly
    if __name__ == "__main__":
        # Make sure features for running multiple processes work correctly when the script is turned into an executable
        autofp.multiprocessing.freeze_support()
        
        # Run the GUI initialization function
        autofp.start_autofp()

        # Execute the application
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
