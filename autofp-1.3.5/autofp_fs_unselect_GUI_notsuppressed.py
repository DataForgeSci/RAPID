import sys
import os
from PyQt4 import QtGui, QtCore
from uiset import Ui
from run import Run
import params
import newversion_set
import thread
import time
import shautofp
import auto
import com
import subauto
import paramgroup

def main():
    # Initialize the Qt application
    app = QtGui.QApplication(sys.argv)

    # Get the working directory
    working_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure com is properly initialized
    com.com_init("ui")

    # Create instances
    r = Run()

    # Ensure all necessary attributes are initialized
    file_path = r''
    r.reset(file_path)

    print("AutoFP GUI is not suppressed. Running refinement by displaying AutoFP GUI.")        
    window = Ui()

    # Make sure com.ui is set to the window instance
    com.ui = window

    # Debug prints to check initialization
    print("com.ui:", com.ui)
    print("window:", window)

    # Load the .pcr file
    window.open(file_path)

    # Perform AUTOSELECT
    window.auto_select()

    # Clear all parameters from tables 0 to 5
    for index in range(6):
        window.tab_refine.setCurrentIndex(index)
        window.table_clear_all()

    # Perform RUN using autorun
    window.autorunfp()

    # Monitor console output for the completion flag ("1 ok!") and automatically close the GUI
    def monitor_console_output():
        output = window.textshow.toPlainText()  # Accessing the console output
        if "1 ok!" in output:
            print("AutoFP Refinement process completed successfully.")
            QtCore.QTimer.singleShot(0, confirm_exit)  # Trigger the close event immediately 

    # Confirm Exit dialog
    def confirm_exit():
        # Find the "Confirm Exit" dialog and click "Yes"
        for widget in QtGui.QApplication.topLevelWidgets():
            if isinstance(widget, QtGui.QMessageBox):
                widget.button(QtGui.QMessageBox.Yes).click()
                break
        window.close()

    # Periodically check the console output
    timer = QtCore.QTimer()
    timer.timeout.connect(monitor_console_output)
    timer.start(0)  # Check for the flag, "1 ok!" immediately 

    # Start the Qt event loop
    sys.exit(app.exec_())
        
if __name__ == '__main__':
    main()
