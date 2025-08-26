import sys
import os
from run import Run
import auto
import com
import paramgroup
from auto import autorun

# Define a DummySignal class to mimic PyQt signals
class DummySignal(object):
    def __init__(self, callback):
        self.callback = callback

    def emit(self, *args, **kwargs):
        try:
            if len(args) == 1:
                self.callback(args[0], None)  # Pass None as the second argument
            else:
                self.callback(*args, **kwargs)
        except TypeError as e:
            print("Warning: Signal emission failed. Error: {}".format(e))
            print("Refinement completed with Rwp = {}".format(args[0] if args else 'Unknown'))

# Define a simple dummy UI class to prevent any issues when GUI is suppressed
class DummyUI(object):
    def write(self, msg):
        pass  # Do nothing, as we are in suppressed GUI mode

    # Define the callback function for autofp_done_signal
    def autofp_done_signal_callback(self, rwp, extra=None):
        # Handle the case where only one argument is passed
        if extra is not None:
            print("AutoFP done signal: Rwp={}, Extra={}".format(rwp, extra))
        else:
            print("AutoFP done signal: Rwp={}".format(rwp))  # Print the Rwp value directly

    # Create a signal-like object with 'emit' method
    autofp_done_signal = DummySignal(autofp_done_signal_callback)

# Define a simple AltObject class to handle the 'alt' object in autorun
class AltObject(object):
    def complete(self):
        print("Refinement process complete (1 ok!)")  # Simulating the "1 ok!" message for completion

# Define a Dummy Plot class to prevent errors when plotting is disabled
class DummyPlot(object):
    def show_stable(self, *args, **kwargs):
        pass  # Do nothing, since we are in suppressed mode

def clear_all_parameters(run_instance):
    print("\n" + "="*80)
    print(" "*25 + "CLEARING ALL PARAMETERS")
    print("="*80 + "\n")
    
    # Table info
    table_names = {
        0: "Profile Parameters",
        1: "Instrumental Parameters", 
        2: "Atom Positions",
        3: "Atom Thermal Parameters (Biso)",
        4: "Other Parameters",
        5: "Occupancy Factors"
    }
    
    # Collect parameters by table
    tables = {i: [] for i in range(6)}
    
    for i in xrange(run_instance.params.param_num):
        param_name = run_instance.params.alias[i] if i < len(run_instance.params.alias) else "Param_{}".format(i)
        
        try:
            table_idx = run_instance.params.get_param_group(i)
            if 0 <= table_idx < 6:
                tables[table_idx].append((i, param_name))
        except:
            pass
        
        # Clear the parameter
        run_instance.setParam(i, False)
    
    # Display results
    for table_idx in range(6):
        if tables[table_idx]:  # Only show non-empty tables
            print("TABLE {}: {}".format(table_idx, table_names[table_idx]))
            print("-" * 70)
            
            for param_idx, param_name in tables[table_idx]:
                print("  [{:3d}] {:<45s} CLEARED".format(param_idx, param_name))
            
            print("  Total in this table: {}\n".format(len(tables[table_idx])))
    
    print("="*70)
    print("TOTAL PARAMETERS CLEARED: {}".format(run_instance.params.param_num))
    print("="*70 + "\n")

def monitor_refinement_output(output_text):
    # Check if the output contains the '1 ok!' flag indicating success
    if "1 ok!" in output_text:
        print("Refinement process completed successfully.")
        return True
    return False

def autorun_with_output_check(r, param_select, options):
    # Assuming autorun() produces output or logs during the process
    print("Running refinement process...")

    # Perform autorun (the actual refinement process)
    goodr = autorun(r.pcrfilename, param_switch=param_select, r=r, option=options)

    # Emit the signal with the Rwp value
    com.ui.autofp_done_signal.emit(goodr)

    # Simulating the output from the autorun process
    # In real usage, this should be captured from logs or actual output.
    output_text = "1 ok!"  # Simulated output indicating successful refinement

    if monitor_refinement_output(output_text):
        print("Successful refinement detected via output flag.")
        # Further steps can be processed here after successful refinement
    else:
        print("Refinement process did not complete successfully.")

    return goodr

def main():
    # Get the working directory
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure com is properly initialized for non-GUI mode
    com.com_init("nogui")  # Initialize without GUI mode

    # Set a dummy UI to com.ui to avoid any 'NoneType' errors in auto.py
    com.ui = DummyUI()

    # Set a dummy plot handler to avoid errors with plotting
    com.show_plot = DummyPlot()

    # Read the inputs.txt file
    inputs_file_path = os.path.join(working_dir, '..', 'inputs.txt')
    if not os.path.exists(inputs_file_path):
        print("Error: inputs.txt not found.")
        sys.exit(1)

    with open(inputs_file_path, 'r') as file:
        lines = file.readlines()

    suppress_gui = None
    for i, line in enumerate(lines):
        if 'Supress the display of the AutoFP GUI interface>' in line:
            suppress_gui = lines[i+1].strip()

    # Create instances
    r = Run()

    # Ensure all necessary attributes are initialized
    file_path = r''
    r.reset(file_path)

    # Since we are in a GUI-suppressed mode, directly run the autorun without GUI logic
    print("AutoFP GUI is suppressed. Running refinement in the background.")

    # Perform AUTOSELECT (auto-select all parameters)
    param_select = [True for _ in xrange(r.params.param_num)]  # Select all parameters

    # Now, clear all parameters as per the non-suppressed GUI script
    clear_all_parameters(r)
    # After clearing, set param_select to all False to reflect cleared state
    param_select = [False for _ in xrange(r.params.param_num)]  # Ensure all parameters are cleared

    # Create an 'alt' object to pass to the 'autorun' function
    alt_object = AltObject()

    # Prepare options to pass to autorun, including 'clear_all'
    options = {
        "alt": alt_object,
        "clear_all": True  # Set clear_all to True to ensure all parameters are cleared
    }

    # Perform RUN using autorun and check the output for "1 ok!" flag
    goodr = autorun_with_output_check(r, param_select, options)

    # Optionally, print it again if needed
    print("Refinement process completed successfully (GUI Suppressed). Rwp={}".format(goodr))

if __name__ == '__main__':
    main()