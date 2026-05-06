#!/usr/bin/bash
# tested only on Ubuntu 24.04

#sudo apt install plantuml        # the UML renderer ✓
#sudo apt install doxygen         # ✓
#sudo apt install graphviz        # ✓
#sudo apt install cflow           # ✓
#sudo apt install python3-pip     # needed for hpp2plantuml, but....

### Ubuntu 24.04 is not happy about "pip install hpp2plantuml"

#sudo apt install pipx     # needed for hpp2plantuml
#pipx install hpp2plantuml
#pipx ensurepath

hpp2plantuml -i "../src/*.hpp" -o astroio.puml
plantuml -tsvg astroio.puml
inkscape --export-type=pdf --export-filename=astroio_class_diagram.pdf --export-area-drawing astroio.svg
