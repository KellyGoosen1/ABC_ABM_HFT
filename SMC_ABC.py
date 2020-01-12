from pyabc import sge
from SMC_ABC_init import abc, \
    smcabc_minimum_epsilon, \
    smcabc_max_nr_populations, temp_folder, \
    version_number, db, smcabc_min_acceptance_rate
import pickle
import os

print(sge.nr_cores_available())

if __name__ == '__main__':

    # Run SMCABC
    history = abc.run(minimum_epsilon=smcabc_minimum_epsilon, max_nr_populations=smcabc_max_nr_populations,
                      min_acceptance_rate=smcabc_min_acceptance_rate)

    # Return True if is ABC history class
    print(history is abc.history)
    print(history.all_runs())

    # Save history object
    with open(os.path.join("/home/gsnkel001/master_dissertation/" + temp_folder, 'SMCABC_history_V' + version_number + '.class'), 'wb') as history_file:
        # Step 3
        pickle.dump(history, history_file)

    with open(os.path.join("/home/gsnkel001/master_dissertation/" + temp_folder, "db.txt"), "w") as text_file:
        print(db, file=text_file)

