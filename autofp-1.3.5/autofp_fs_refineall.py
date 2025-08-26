import sys
import os
from PyQt4 import QtGui, QtCore
from uiset import Ui
from run import Run
import params
import newversion_set
import thread
import time
#import multiprocessing
import shautofp
import auto
import com

def main():
    # Initialize the Qt application
    app = QtGui.QApplication(sys.argv)

    # Ensure com is properly initialized
    com.com_init("ui")

    # Create instances
    r = Run()
    window = Ui()

    # Make sure com.ui is set to the window instance
    com.ui = window

    # Debug prints to check initialization
    print("com.ui: {}".format(com.ui))
    print("window: {}".format(window))

    # Load the .pcr file
    file_path = r''
    window.open(file_path)

    # Perform AUTOSELECT
    window.auto_select()

    # Debug prints to check initialization and settings
    print("com.ui.ui: {}".format(com.ui.ui))
    print("com.ui.ui.check_show_rwp.isChecked(): {}".format(com.ui.ui.check_show_rwp.isChecked()))

    # Ensure `plot` is imported before using threading
    print("Importing plot module to ensure availability for threading.")
    import plot

    # Perform RUN using autorun
    print("Starting autorunfp...")
    window.autorunfp()
    print("Finished autorunfp...")

    # Start the Qt event loop
    print("Starting Qt event loop...")
    sys.exit(app.exec_())
    print("Qt event loop has started.")  # This will not be executed if app.exec_() runs correctly

if __name__ == '__main__':
    main()
