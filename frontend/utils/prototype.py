import os
    

def lookup_checkpoint_files():

    flie_list = list(os.listdir('/opt/ml/final_project/web/'))
    flie_list.sort()
    checkpoint_flie_list = []
    for file in flie_list:
        if file[-3:] == '.pt':
            checkpoint_flie_list.append(file)

    return tuple(checkpoint_flie_list)
