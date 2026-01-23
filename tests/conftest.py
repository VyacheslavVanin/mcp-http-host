import sys
# Patch command line arguments to avoid argparse issues during import
sys.argv = ["test"]