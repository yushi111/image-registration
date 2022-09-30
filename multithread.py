import _thread

from directDr import *

try:
    _thread.start_new_thread(main_run,(0,18,0,))
    _thread.start_new_thread(main_run,(18,36,1,))
    _thread.start_new_thread(main_run,(36,54,2,))
    _thread.start_new_thread(main_run,(54,70,3,))
except:
    print('thread failed')
