from dc_removal_python import main as python_main
from dc_removal.dc_rem_specializer import main as sejits_main
from numpy import array_equal

python_result = python_main()
sejits_result = sejits_main()

print "Match: ", array_equal(python_result, sejits_result)
