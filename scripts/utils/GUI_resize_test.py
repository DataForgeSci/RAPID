import sys
from PyQt4 import QtGui

class TestExample(QtGui.QWidget):
    def __init__(self, parent=None):
        super(TestExample, self).__init__(parent)
        
        self.initUI()
        
    def initUI(self):
        open_button = QtGui.QPushButton('OPEN', self)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(open_button)
        self.setLayout(layout)
        
        self.setMinimumSize(200, 150)
        
        self.setWindowTitle('Test Example')
        self.setGeometry(300, 300, 300, 200)

def main():
    app = QtGui.QApplication(sys.argv)
    
    window = TestExample()
    
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
