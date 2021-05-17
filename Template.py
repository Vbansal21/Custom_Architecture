import time

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if abs(sec) < 60:
            return str(int(sec)) + " sec"
        elif abs(sec) < (60*60):
            return str(int(sec/60)) + " min " + str(int(sec%60)) + " sec"
        else:
            return str(int(sec/(60*60)))+" hr "+str(int((sec%3600)/60))+" min "+str(int((sec%3600)%60))+" sec"
    def started_time(self):
        return self.start_time
    def elapsed_sec(self):
        return time.time()-self.start_time
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))

total_elapsed_time = ElapsedTimer()

import datetime
import os
#os.system("pip install inputimeout nvidia-ml-py3 matplotlib numpy tensorflow-gpu tensorflow-datasets tfds-nightly")

from inputimeout import inputimeout as inpt

date_time_at_start = datetime.datetime.now()

dir_path = os.path.dirname(os.path.realpath(__file__))
print("Your current directory of this python file execution is:",dir_path)

try:
    logging_files = open("Name_of_log_files.txt",'r+')
except:
    logging_files = open("Name_of_log_files.txt",'w')
    logging_files = open("Name_of_log_files.txt",'r+')
list_of_file_names = []
for file_names in logging_files:
    list_of_file_names.append(str(file_names)[:-1])

print("Previously used files:",list_of_file_names)

log_path = os.path.dirname(os.path.realpath(__file__))

same_dir = "Yes"
try:
    same_dir = inpt(prompt="Should the logging file be placed in current dir? (Yes/No):",timeout=15)
    
    if (same_dir=="No" or same_dir=="no" or same_dir=="N" or same_dir=="n" or same_dir=='0'):
        log_path = inpt(prompt="Directory Name:",timeout=60)
except:
    pass

file_name = log_path + "/Log.txt"
temp = ""

try:
    temp = inpt(prompt="Name for error Logging file(Default is: 'Log.txt'. Leave empty if need to remain default.):",timeout=15)
    if len(temp) >= 1:
        file_name = log_path + "/" + str(temp) + ".txt"
except:
    pass

if file_name not in list_of_file_names:
    logging_files.write(file_name)
    logging_files.write('\n')
  
try:
    pass
except Exception as e:
    try:
        f = open(file_name,'r+')
    except:
        f = open(file_name,'w')
        f = open(file_name,'r+')
    last_count = None
    try:
        for line in f:
            last_count = line
        last_count = int(last_count)
    except:
        last_count = 0
    last_count += 1
    f.write(str(date_time_at_start))
    f.write("{")
    f.write(str(e))
    f.write("}\n")
    f.write(str(last_count))
    f.write("\n")
    print("Total no. of errors done since last reset:",last_count)
    print("To check the errors, see the file:",file_name)
finally:
    total_elapsed_time.elapsed_time()
